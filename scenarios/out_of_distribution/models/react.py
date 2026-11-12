from abc import abstractmethod

import torch
import torch.nn.functional as F
from torch import nn


class BaseModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len=12, horizon=12):
        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon

    @abstractmethod
    def forward(self):  # (x, label=None, adj=None)
        raise NotImplementedError

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])


def resolve_temporal_index(values, size):
    if values.max() <= 1.0 + 1e-6:
        values = values * size
    return values.long().clamp_(0, size - 1)


def select_node_rows(tensor, node_indices, node_num):
    if tensor is None:
        return None
    if node_indices is None:
        return tensor[:node_num]
    index = torch.as_tensor(node_indices, device=tensor.device, dtype=torch.long)
    return tensor.index_select(0, index)


def compute_context_stats(raw_series, tod_norm):
    mean = raw_series.mean(dim=-1)
    std = torch.sqrt(raw_series.var(dim=-1, unbiased=False) + 1e-5)
    last = raw_series[..., -1]
    trend = last - raw_series[..., 0]
    if raw_series.size(-1) > 1:
        volatility = (raw_series[..., 1:] - raw_series[..., :-1]).abs().mean(dim=-1)
    else:
        volatility = torch.zeros_like(mean)
    return torch.stack([mean, std, last, trend, volatility, tod_norm], dim=-1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.10):
        super().__init__()
        self.fc1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=float(dropout))

    def forward(self, x):
        hidden = self.fc2(self.drop(self.act(self.fc1(x))))
        return x + hidden


class React(BaseModel):
    """React model for spatiotemporal forecasting.

    Architecture:
    - base path: conv over the full input window
    - base mixer: CSF-style ResidualMLPBlock
    - context path: fixed stats6 + optional static prior + prior-keep indicator
    - adaptation: uncertainty-gated support + FiLM/gate
    """

    def __init__(
        self,
        node_num,
        input_dim,
        output_dim,
        seq_len=12,
        horizon=12,
        embed_dim=48,
        temp_dim_tid=32,
        temp_dim_diw=32,
        num_layer=1,
        mlp_dropout=0.10,
        steps_per_day=96,
        days_per_week=7,
        use_tod_embedding=True,
        use_dow_embedding=True,
        static_prior=None,
        adj_mx=None,
        context_dim=32,
        context_hidden=64,
        support_mode="entropy",
        support_mix=0.10,
        support_mix_min=0.05,
        context_reg_weight=0.0,
        use_context=True,
        use_dynamic_context=True,
        use_context_gate=True,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            horizon=horizon,
        )
        self.node_num = int(node_num)
        self.time_of_day_size = int(steps_per_day)
        self.day_of_week_size = int(days_per_week)
        self.use_tod_embedding = bool(use_tod_embedding)
        self.use_dow_embedding = bool(use_dow_embedding)
        self.support_mode = str(support_mode).lower().strip()
        self.support_mix = float(support_mix)
        self.support_mix_min = float(support_mix_min)
        self.context_reg_weight = float(context_reg_weight)
        self.use_context = bool(use_context)
        self.use_dynamic_context = bool(use_dynamic_context)
        self.use_context_gate = bool(use_context_gate)

        if self.output_dim != 1:
            raise ValueError("React currently expects output_dim=1.")
        if self.support_mode not in ("none", "fixed", "entropy"):
            raise ValueError(f"Unsupported support_mode: {support_mode}")

        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.seq_len,
            out_channels=embed_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        self.time_in_day_emb = None
        if self.use_tod_embedding:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, temp_dim_tid)
            )
            nn.init.xavier_uniform_(self.time_in_day_emb)

        self.day_in_week_emb = None
        if self.use_dow_embedding:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, temp_dim_diw)
            )
            nn.init.xavier_uniform_(self.day_in_week_emb)

        self.hidden_dim = embed_dim
        if self.use_tod_embedding:
            self.hidden_dim += temp_dim_tid
        if self.use_dow_embedding:
            self.hidden_dim += temp_dim_diw

        self.encoder = nn.Sequential(
            *[
                ResidualMLPBlock(self.hidden_dim, dropout=mlp_dropout)
                for _ in range(int(num_layer))
            ]
        )
        self.base_head = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.horizon * self.output_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        if static_prior is None:
            self.static_prior = None
            prior_dim = 0
        else:
            self.register_buffer("static_prior", static_prior.float())
            prior_dim = int(static_prior.shape[-1])

        if adj_mx is None or self.support_mode == "none":
            self.adj_normed = None
        else:
            adj_t = adj_mx.float()
            row_sum = adj_t.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            self.register_buffer("adj_normed", adj_t / row_sum)

        self.state_dim = 6 + prior_dim + 1
        self.context_mlp = nn.Sequential(
            nn.Linear(self.state_dim, context_hidden),
            nn.GELU(),
            nn.Linear(context_hidden, context_dim),
        )
        self.context_norm = nn.LayerNorm(context_dim)
        self.router = nn.Linear(context_dim, 4)

        self.context_to_hidden = nn.Sequential(
            nn.Linear(context_dim, context_hidden),
            nn.GELU(),
            nn.Linear(context_hidden, 2 * self.hidden_dim),
        )
        self.hidden_norm = nn.LayerNorm(self.hidden_dim)
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.context_gate = None
        if self.use_context_gate:
            self.context_gate = nn.Sequential(
                nn.Linear(self.hidden_dim + context_dim, context_hidden),
                nn.GELU(),
                nn.Linear(context_hidden, self.hidden_dim),
                nn.Sigmoid(),
            )

    def encode_series(self, input_data):
        batch_size, _, num_nodes, _ = input_data.shape
        series_features = (
            input_data.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)
        )
        return self.time_series_emb_layer(series_features.transpose(1, 2).unsqueeze(-1))

    def build_base_hidden(self, input_data, tod_norm, dow_norm):
        feature_blocks = [self.encode_series(input_data)]

        if self.use_tod_embedding:
            tod_idx = resolve_temporal_index(tod_norm, self.time_of_day_size)
            feature_blocks.append(
                self.time_in_day_emb[tod_idx].transpose(1, 2).unsqueeze(-1)
            )

        if self.use_dow_embedding:
            dow_idx = resolve_temporal_index(dow_norm, self.day_of_week_size)
            feature_blocks.append(
                self.day_in_week_emb[dow_idx].transpose(1, 2).unsqueeze(-1)
            )

        hidden = torch.cat(feature_blocks, dim=1)
        return self.encoder(hidden)

    def build_context_state(self, raw_series, tod_norm, node_indices, prior_mask):
        context_stats = compute_context_stats(raw_series, tod_norm)
        if not self.use_dynamic_context:
            context_stats = torch.zeros_like(context_stats)
        batch_size, node_num, _ = context_stats.shape
        state_parts = [context_stats]

        prior_keep_indicator = context_stats.new_ones(batch_size, node_num, 1)
        if prior_mask is not None:
            keep = prior_mask.to(context_stats.device).float().view(1, node_num, 1)
            prior_keep_indicator = keep.expand(batch_size, -1, -1)

        if self.static_prior is not None:
            static_prior = select_node_rows(
                self.static_prior, node_indices, node_num
            ).to(context_stats.device)
            static_prior = static_prior.unsqueeze(0).expand(batch_size, -1, -1)
            if prior_mask is not None:
                static_prior = static_prior * prior_keep_indicator
            state_parts.append(static_prior)

        state_parts.append(prior_keep_indicator)
        return torch.cat(state_parts, dim=-1)

    def select_adj_submatrix(self, node_indices):
        if self.adj_normed is None:
            return None
        if node_indices is None:
            return self.adj_normed
        index = torch.as_tensor(
            node_indices, device=self.adj_normed.device, dtype=torch.long
        )
        return self.adj_normed[index][:, index]

    def stabilize_context(self, raw_context, node_indices):
        routing_logits = self.router(raw_context)
        routing_probs = F.softmax(routing_logits, dim=-1)
        eps = 1e-8
        routing_entropy = -(routing_probs * (routing_probs + eps).log()).sum(
            dim=-1, keepdim=True
        )
        denom = torch.log(
            routing_probs.new_tensor(float(routing_probs.shape[-1]))
        ).clamp_min(eps)
        routing_entropy = routing_entropy / denom

        adj_submatrix = self.select_adj_submatrix(node_indices)
        if adj_submatrix is None:
            support_alpha = raw_context.new_zeros(
                raw_context.shape[0], raw_context.shape[1], 1
            )
            return raw_context, routing_entropy, support_alpha

        supported_context = torch.einsum("mn,bnd->bmd", adj_submatrix, raw_context)
        if self.support_mode == "fixed":
            support_alpha = raw_context.new_full(
                (raw_context.shape[0], raw_context.shape[1], 1), self.support_mix
            )
        else:
            support_alpha = (
                self.support_mix_min
                + (self.support_mix - self.support_mix_min) * routing_entropy
            )
        support_alpha = support_alpha.clamp_(0.0, 1.0)

        stabilized_context = (
            1.0 - support_alpha
        ) * raw_context + support_alpha * supported_context
        return self.context_norm(stabilized_context), routing_entropy, support_alpha

    def apply_context_adaptation(self, base_hidden, context):
        film_params = self.context_to_hidden(context)
        scale, bias = torch.chunk(film_params, chunks=2, dim=-1)

        normalized_hidden = self.hidden_norm(base_hidden)
        hidden_delta = normalized_hidden * torch.tanh(scale) + bias
        if self.context_gate is not None:
            adaptation_gate = self.context_gate(
                torch.cat([normalized_hidden, context], dim=-1)
            )
            hidden_delta = adaptation_gate * hidden_delta
        return self.output_norm(base_hidden + hidden_delta)

    def forward(self, x, node_indices=None, prior_mask=None):
        input_data = x[..., : self.input_dim]
        tod_norm = x[:, -1, :, 1]
        dow_norm = x[:, -1, :, 2]
        raw_series = x[:, :, :, 0].transpose(2, 1).contiguous()

        hidden_4d = self.build_base_hidden(input_data, tod_norm, dow_norm)
        base_hidden = hidden_4d.squeeze(-1).transpose(1, 2).contiguous()

        if not self.use_context:
            pred = self.base_head(
                base_hidden.transpose(1, 2).unsqueeze(-1).contiguous()
            )
            zero = pred.new_tensor(0.0)
            return {
                "pred": pred,
                "context_entropy": zero,
                "support_alpha": zero,
                "context_reg": zero,
            }

        context_state = self.build_context_state(
            raw_series, tod_norm, node_indices, prior_mask
        )
        raw_context = self.context_norm(self.context_mlp(context_state))
        context, context_entropy, support_alpha = self.stabilize_context(
            raw_context, node_indices
        )

        adapted_hidden = self.apply_context_adaptation(base_hidden, context)
        pred = self.base_head(adapted_hidden.transpose(1, 2).unsqueeze(-1).contiguous())

        if self.context_reg_weight > 0:
            context_reg = self.context_reg_weight * support_alpha.pow(2).mean()
        else:
            context_reg = pred.new_tensor(0.0)

        return {
            "pred": pred,
            "context_entropy": context_entropy.mean(),
            "support_alpha": support_alpha.mean(),
            "context_reg": context_reg,
        }
