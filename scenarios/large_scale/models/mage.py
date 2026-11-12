import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAGEWrapper(nn.Module):
    """Adapter that exposes MAGE as model(NormX, TE) for this pipeline."""

    def __init__(
        self,
        node_num,
        seq_len,
        horizon,
        tod,
        dow,
        model_dim=128,
        recur_num=16,
        topk=4,
        graph_gen_dim=32,
        ffn_dropout=0.3,
    ):
        super().__init__()
        if topk > recur_num:
            raise ValueError(
                f"MAGE topk ({topk}) cannot exceed recur_num ({recur_num})."
            )

        self.node_num = node_num
        self.seq_len = seq_len
        self.horizon = horizon
        self.tod = tod
        self.dow = dow

        args = SimpleNamespace(
            node_num=node_num,
            seq_len=seq_len,
            horizon=horizon,
            input_dim=3,
            output_dim=1,
            feature_dim=1,
            model_dim=model_dim,
            recur_num=recur_num,
            topk=topk,
            graph_gen_dim=graph_gen_dim,
            ffn_dropout=ffn_dropout,
            second=0,
            minute=tod,
            hour=0,
            day=dow,
            week=0,
            weekday=0,
            month=0,
            quarter=0,
            year=0,
        )
        self.core = MAGECore(args)

    def forward(self, x, te):
        if te.dim() == 3:
            te = te.unsqueeze(2).expand(-1, -1, x.shape[2], -1)
        tod_feature = te[..., 0:1].float() / float(self.tod)
        dow_feature = te[..., 1:2].float() / float(self.dow)
        mage_x = torch.cat([x.float(), tod_feature, dow_feature], dim=-1)
        return self.core(mage_x)

    def param_num(self):
        return sum(param.numel() for param in self.parameters())


class MAGECore(nn.Module):
    """Mixture of Adaptive Graph Experts adapted from PoorOtterBob/MAGE."""

    def __init__(self, model_args):
        super().__init__()
        self.args = model_args
        self.node_num = model_args.node_num
        self.seq_len = model_args.seq_len
        self.horizon = model_args.horizon
        self.input_dim = model_args.input_dim
        self.output_dim = model_args.output_dim

        self.encoder = nn.Linear(self.seq_len * self.input_dim, self.args.model_dim)
        self.decoder = nn.Linear(self.args.model_dim, self.horizon * self.output_dim)
        self.decoder2 = nn.Linear(self.args.model_dim, self.horizon * self.output_dim)

        self.prompt = PromptPool(self.args)
        self.backbone1 = PreNormTransformer(
            self.args.model_dim,
            self.args.recur_num,
            self.node_num,
            blocknum=1,
            topk=self.args.topk,
            graph_gen_dim=self.args.graph_gen_dim,
            ffn_dropout=self.args.ffn_dropout,
        )
        self.backbone2 = PreNormTransformer(
            self.args.model_dim,
            self.args.recur_num,
            self.node_num,
            blocknum=2,
            topk=self.args.topk,
            graph_gen_dim=self.args.graph_gen_dim,
            ffn_dropout=self.args.ffn_dropout,
        )
        self.backbone3 = PreNormTransformer(
            self.args.model_dim,
            self.args.recur_num,
            self.node_num,
            blocknum=3,
            topk=self.args.topk,
            graph_gen_dim=self.args.graph_gen_dim,
            ffn_dropout=self.args.ffn_dropout,
        )
        self.register_buffer("routing_bias", torch.zeros(3, self.args.recur_num))

    def forward(self, x, label=None, bias=None, return_load=False):
        if bias is None:
            bias = self.routing_bias

        prompt = self.prompt(x)
        h = x = self.encoder(self.channel_compress(x)) + prompt
        z, topk_indices1 = self.backbone1(x, bias[0])
        x, topk_indices2 = self.backbone2(z, bias[1])
        x, topk_indices3 = self.backbone3(h - x, bias[2])
        x = self.decoder(x) + self.decoder2(z)
        pred = self.channel_decompress(x)

        if self.training and return_load:
            load = torch.stack(
                [
                    torch.bincount(
                        topk_indices1.flatten(), minlength=self.args.recur_num
                    ),
                    torch.bincount(
                        topk_indices2.flatten(), minlength=self.args.recur_num
                    ),
                    torch.bincount(
                        topk_indices3.flatten(), minlength=self.args.recur_num
                    ),
                ],
                dim=0,
            )
            return pred, load
        return pred

    def channel_compress(self, x):
        batch_size = x.shape[0]
        return x.transpose(1, 2).reshape(batch_size, self.node_num, -1)

    def channel_decompress(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, self.node_num, self.horizon, -1).transpose(1, 2)


class PreNormTransformer(nn.Module):
    def __init__(
        self,
        dim,
        expert_num,
        node_num,
        blocknum,
        topk=4,
        graph_gen_dim=32,
        ffn_dropout=0.3,
    ):
        super().__init__()
        self.attn = MAGEBlock(
            dim,
            expert_num,
            node_num,
            depth=blocknum,
            model_dim=dim,
            graph_gen_dim=graph_gen_dim,
            topk=topk,
        )
        self.attn_norm = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, dim, dropout=ffn_dropout)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x, bias):
        x_res = x
        x, topk_indices = self.attn(self.attn_norm(x), bias)
        x = x + x_res
        x = self.ffn(self.ffn_norm(x)) + x
        return x, topk_indices


class MAGEBlock(nn.Module):
    def __init__(
        self, dim, expert_num, node_num, depth, model_dim, graph_gen_dim=32, topk=4
    ):
        super().__init__()
        self.E1 = nn.Parameter(torch.rand(2, expert_num, node_num, graph_gen_dim))
        self.E2 = nn.Parameter(torch.rand(2, expert_num, graph_gen_dim, node_num))
        self.router = nn.Linear(dim, expert_num)

        self.outer1 = nn.Parameter(torch.empty((dim, dim)).normal_(mean=0, std=0.1))
        self.outer2 = nn.Parameter(torch.eye(dim), requires_grad=False)
        self.outer_lambda = nn.Parameter(torch.tensor(99.0).log())

        self.depth = depth
        self.lambda_init = self.lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(
            torch.empty(model_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.empty(model_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.empty(model_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.empty(model_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )

        self.coefficient = nn.Parameter(torch.ones(1), requires_grad=False)
        self.topk = topk

    @staticmethod
    def lambda_init_fn(depth):
        return 0.8 - 0.6 * math.exp(-0.3 * depth)

    def cal_lambda(self):
        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1, keepdim=True).float()
        )
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1, keepdim=True).float()
        )
        return -(lambda_1 - lambda_2 + self.lambda_init)

    def cal_coefficient(self):
        return torch.cat([self.coefficient, self.cal_lambda()], dim=0)

    def cal_outer(self):
        outer_weight = torch.sigmoid(self.outer_lambda)
        return outer_weight * self.outer2 + (1 - outer_weight) * self.outer2

    def selector(self, x, bias=0):
        route = self.router(x)
        _, topk_indices = torch.topk(torch.sigmoid(route) + bias, self.topk, dim=-1)
        mask = torch.zeros_like(route, dtype=torch.bool).scatter_(-1, topk_indices, 1)
        route_mask = self.normer(torch.sigmoid(route + mask.log()))
        return route_mask, topk_indices

    @staticmethod
    def normer(route):
        return route / torch.sum(route, dim=-1, keepdim=True).clamp_min(1e-12)

    def forward(self, x, bias):
        route, topk_indices = self.selector(x, bias)
        x = torch.einsum(
            "bik,e,ekid,ekdj,bjf,fc->bic",
            route,
            self.cal_coefficient(),
            torch.softmax(self.E1, dim=-1),
            torch.softmax(self.E2, dim=-1),
            x,
            self.cal_outer(),
        )
        return x, topk_indices


class SwiGLUFFN(nn.Module):
    def __init__(self, dim_in, dim_out, expand_ratio=4, dropout=0.3):
        super().__init__()
        self.W1 = nn.Linear(dim_in, expand_ratio * dim_in)
        self.W2 = nn.Linear(dim_in, expand_ratio * dim_in)
        self.W3 = nn.Linear(expand_ratio * dim_in, dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.W3(self.dropout(F.silu(self.W1(x)) * self.W2(x)))


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1.0, eps=1e-8, bias=False):
        super().__init__()
        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias
        self.scale = nn.Parameter(torch.ones(d))
        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)
            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-0.5)
        x_normed = x / (rms_x + self.eps)
        if self.bias:
            return self.scale * x_normed + self.offset
        return self.scale * x_normed


class PromptPool(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.feature_dim = model_args.feature_dim
        self.spatial_prompt = nn.Parameter(
            torch.empty(model_args.node_num, model_args.model_dim)
        )
        nn.init.xavier_uniform_(self.spatial_prompt)

        prompt_specs = [
            ("second", model_args.second),
            ("minute", model_args.minute),
            ("hour", model_args.hour),
            ("day", model_args.day),
            ("week", model_args.week),
            ("weekday", model_args.weekday),
            ("month", model_args.month),
            ("quarter", model_args.quarter),
            ("year", model_args.year),
        ]
        self.pool = nn.ModuleList()
        self.denorm = []
        for _, size in prompt_specs:
            if size:
                self.pool.append(nn.Embedding(size, model_args.model_dim))
                self.denorm.append(size)

    def forward(self, temporal):
        prompt = 0
        for i, embedding in enumerate(self.pool):
            denorm = self.denorm[i]
            index = torch.round(
                temporal[:, -1, :, i + self.feature_dim] * denorm
            ).long()
            index = index.clamp(0, denorm - 1)
            prompt = prompt + embedding(index)
        return prompt + self.spatial_prompt
