import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMixer(nn.Module):
    """Feature mixer used by the hierarchical linear interaction block."""

    def __init__(self, in_features, hidden_features, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class HLI(nn.Module):
    """Hierarchical Linear Interaction block.

    Each block performs Inter-Patch Communication followed by
    Intra-Patch Refinement.
    """

    def __init__(self, hidden_size, num, size, rank, mlp_ratio=4.0):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num, self.size = num, size

        # Inter-Patch Communication
        self.inter_patch_communication_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.inter_patch_communication = FeatureMixer(hidden_size, hidden_size)
        self.inter_patch_communication_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.inter_patch_feature_mixer = FeatureMixer(hidden_size, mlp_hidden_dim)

        # Intra-Patch Refinement
        self.intra_patch_refinement_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.intra_patch_refinement = FeatureMixer(hidden_size, hidden_size)
        self.intra_patch_refinement_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.intra_patch_feature_mixer = FeatureMixer(hidden_size, mlp_hidden_dim)

        self.low_proj1 = nn.Linear(num, rank, bias=False)
        self.low_proj2 = nn.Linear(rank, num, bias=False)

    def forward(self, x):
        # Input: x [B, T, P*N, D] -> [B, T, P, N, D]
        B, T, _, D = x.shape
        P, N = self.num, self.size  # P: patch num, N: patch size
        assert self.num * self.size == _
        x = x.reshape(B, T, P, N, D)

        # Inter-Patch Communication: [B, T, P, N, D] -> [B*T*N, P, D]
        x_inter_patch = x.transpose(2, 3).reshape(B * T * N, P, D)  # (B*T*N, P, D)
        x_inter_patch_norm = self.inter_patch_communication_norm(x_inter_patch)
        inter_patch_out = self.inter_patch_communication(
            x_inter_patch_norm
        )  # (B*T*N, P, D)

        # Projection: [B*T*N, P, D] -> [B*T*N, D, P] -> [B*T*N, P, D]
        inter_patch_out_t = inter_patch_out.transpose(1, 2)  # (B*T*N, D, P)
        inter_patch_out_proj = self.low_proj1(inter_patch_out_t)
        inter_patch_out_proj = self.low_proj2(inter_patch_out_proj)
        inter_patch_out = inter_patch_out_proj.transpose(1, 2)  # (B*T*N, P, D)

        # Reshape back: [B*T*N, P, D] -> [B, T, P, N, D]
        x = x + inter_patch_out.reshape(B, T, N, P, D).transpose(2, 3)
        x = x + self.inter_patch_feature_mixer(self.inter_patch_communication_norm2(x))

        # Intra-Patch Refinement: [B, T, P, N, D] -> [B*T*P, N, D]
        x_intra_patch = x.reshape(B * T * P, N, D)  # (B*T*P, N, D)
        x_intra_patch_norm = self.intra_patch_refinement_norm(x_intra_patch)
        intra_patch_out = self.intra_patch_refinement(
            x_intra_patch_norm
        )  # (B*T*P, N, D)
        x = x + intra_patch_out.reshape(
            B, T, P, N, D
        )  # [B*T*P, N, D] -> [B, T, P, N, D]
        x = x + self.intra_patch_feature_mixer(self.intra_patch_refinement_norm2(x))

        # Output: [B, T, P, N, D] -> [B, T, P*N, D]
        return x.reshape(B, T, -1, D)


class SqLinear(nn.Module):
    """Spatial-temporal traffic forecasting model with hierarchical linear interactions"""

    def __init__(
        self,
        tem_patchsize,
        tem_patchnum,
        node_num,
        spa_patchsize,
        spa_patchnum,
        tod,
        dow,
        layers,
        input_dims,
        node_dims,
        tod_dims,
        dow_dims,
        ori_parts_idx,
        reo_parts_idx,
        reo_all_idx,
        output_len=12,
        rank=16,
        mlp_ratio=1.0,
    ):
        super(SqLinear, self).__init__()
        self.node_num = node_num
        self.tod, self.dow = tod, dow
        self.tem_patchsize = tem_patchsize

        self.register_buffer(
            "ori_parts_idx",
            torch.as_tensor(ori_parts_idx, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "reo_parts_idx",
            torch.as_tensor(reo_parts_idx, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "reo_all_idx",
            torch.as_tensor(reo_all_idx, dtype=torch.long),
            persistent=False,
        )

        # Model dimensions
        dims = input_dims + tod_dims + dow_dims + node_dims

        # Spatio-temporal embedding layers
        self.input_st_fc = nn.Conv2d(
            in_channels=3,
            out_channels=input_dims,
            kernel_size=(1, tem_patchsize),
            stride=(1, tem_patchsize),
            bias=True,
        )
        self.node_emb = nn.Parameter(torch.empty(node_num, node_dims))
        nn.init.xavier_uniform_(self.node_emb)
        self.time_in_day_emb = nn.Parameter(torch.empty(tod, tod_dims))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(torch.empty(dow, dow_dims))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        # Hierarchical linear interaction encoder
        self.spa_encoder = nn.ModuleList(
            [
                HLI(dims, spa_patchnum, spa_patchsize, rank, mlp_ratio=mlp_ratio)
                for _ in range(layers)
            ]
        )
        # Projection decoder
        self.regression_conv = nn.Conv2d(
            in_channels=tem_patchnum * dims,
            out_channels=output_len,
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(self, x, te):
        # x: [B,T,N,1] input traffic
        # te: [B,T,2] or [B,T,N,2] time information

        # Spatio-temporal embedding
        embeded_x = self.embedding(x, te)
        rex = embeded_x.index_select(2, self.reo_all_idx)  # select patched points

        # Hierarchical linear interaction encoding
        for block in self.spa_encoder:
            rex = block(rex)

        # Restore original spatial structure
        orginal = x.new_zeros(rex.shape[0], rex.shape[1], self.node_num, rex.shape[-1])
        orginal.index_copy_(
            2, self.ori_parts_idx, rex.index_select(2, self.reo_parts_idx)
        )

        # Projection decoding
        pred_y = self.regression_conv(
            orginal.transpose(2, 3).reshape(orginal.shape[0], -1, orginal.shape[-2], 1)
        )

        return pred_y  # [B, output_len, N, 1]

    def embedding(self, x, te):
        b, _, n, _ = x.shape
        if te.dim() == 3:
            te_steps = te
            te_features = te.unsqueeze(2).expand(-1, -1, n, -1)
        else:
            te_steps = te[:, :, 0, :]
            te_features = te

        # Combine input traffic with time features
        x1 = torch.cat(
            [
                x,
                (te_features[..., 0:1].to(x.dtype) / self.tod),
                (te_features[..., 1:2].to(x.dtype) / self.dow),
            ],
            -1,
        )
        input_data = self.input_st_fc(x1.transpose(1, 3).contiguous()).transpose(1, 3)
        t_patch = input_data.shape[1]

        patch_indices = torch.arange(
            self.tem_patchsize - 1,
            self.tem_patchsize * t_patch,
            self.tem_patchsize,
            device=te_steps.device,
        )

        # Add time of day embedding
        t_i_d_data = te_steps.index_select(1, patch_indices)[:, :t_patch, 0]
        tod_emb = (
            self.time_in_day_emb[t_i_d_data.long()].unsqueeze(2).expand(-1, -1, n, -1)
        )
        input_data = torch.cat([input_data, tod_emb], -1)

        # Add day of week embedding
        d_i_w_data = te_steps.index_select(1, patch_indices)[:, :t_patch, 1]
        dow_emb = (
            self.day_in_week_emb[d_i_w_data.long()].unsqueeze(2).expand(-1, -1, n, -1)
        )
        input_data = torch.cat([input_data, dow_emb], -1)

        # Add spatial embedding
        node_emb = self.node_emb.unsqueeze(0).unsqueeze(1).expand(b, t_patch, -1, -1)
        input_data = torch.cat([input_data, node_emb], -1)

        return input_data

    def get_n_param(self):
        """Count trainable parameters"""
        n_param = 0
        for param in self.parameters():
            if param.requires_grad:
                n_param += torch.numel(param)
        return n_param
