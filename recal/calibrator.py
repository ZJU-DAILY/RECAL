"""Low-rank horizon/node calibrator and forecast-space helpers for RECAL.

The calibrator is the *state* maintained by the streaming forecast-maintenance
operators in :mod:`recal.maintenance`.  Its low-rank loading matrix
``horizon_node`` (``U`` in the paper) is shared by the multiplicative and
additive surfaces; it doubles as the free node-importance value function used by
the budgeted Top-K maintenance operator.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


TensorLoss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class ReCalibrator(nn.Module):
    """Low-rank horizon/node calibrator used by RECAL.

    Input and output use the normalized forecast layout ``[B, 1, N, H]``.
    The horizon domain is a rank-``r`` outer product ``horizon_node @ horizon_*``
    (shared spatial loading ``U`` of shape ``[N, r]``); the node domain is a
    per-node affine tail.
    """

    def __init__(
        self,
        num_nodes: int,
        pred_len: int,
        rank: int = 4,
        use_horizon_domain: bool = True,
        use_node_domain: bool = True,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.pred_len = int(pred_len)
        self.rank = int(rank)
        self.use_horizon_domain = bool(use_horizon_domain)
        self.use_node_domain = bool(use_node_domain)

        if self.use_horizon_domain:
            self.horizon_node = nn.Parameter(torch.empty(self.num_nodes, self.rank))
            self.horizon_scale = nn.Parameter(torch.zeros(self.rank, self.pred_len))
            self.horizon_shift = nn.Parameter(torch.zeros(self.rank, self.pred_len))
            nn.init.normal_(self.horizon_node, mean=0.0, std=1e-3)
        else:
            self.register_parameter("horizon_node", None)
            self.register_parameter("horizon_scale", None)
            self.register_parameter("horizon_shift", None)

        if self.use_node_domain:
            self.node_scale = nn.Parameter(torch.zeros(self.num_nodes, 1))
            self.node_shift = nn.Parameter(torch.zeros(self.num_nodes, 1))
        else:
            self.register_parameter("node_scale", None)
            self.register_parameter("node_shift", None)

    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
        y = y_pred[:, 0]
        if self.use_horizon_domain:
            scale = torch.matmul(self.horizon_node, self.horizon_scale)
            shift = torch.matmul(self.horizon_node, self.horizon_shift)
            y = y * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
        if self.use_node_domain:
            y = y * (1 + self.node_scale.unsqueeze(0)) + self.node_shift.unsqueeze(0)
        return y.unsqueeze(1)

    def forward_selected(
        self, raw: torch.Tensor, horizons: torch.Tensor
    ) -> torch.Tensor:
        """Calibrate selected raw horizons [K, N] without K x N x H activations.

        Keep the original N x H GEMMs and affine order. This is an execution
        alternative, not a different calibrator; floating-point gradients need
        not be bitwise identical to full-broadcast backward.
        """
        y = raw
        if self.use_horizon_domain:
            scale = (
                (self.horizon_node @ self.horizon_scale).t().index_select(0, horizons)
            )
            shift = (
                (self.horizon_node @ self.horizon_shift).t().index_select(0, horizons)
            )
            y = y * (1 + scale) + shift
        if self.use_node_domain:
            y = y * (1 + self.node_scale.t()) + self.node_shift.t()
        return y

    def node_loading_norm(self) -> torch.Tensor:
        """Per-node loading norm ``||U[i, :]||_2`` of shape ``[N]``.

        This is the *free* node-importance signal exploited by budgeted Top-K
        maintenance (no extra parameters): nodes with a larger response to the
        shared drift field carry more of the calibration energy.
        """
        if self.use_horizon_domain and self.horizon_node is not None:
            base = torch.linalg.norm(self.horizon_node, dim=1)
        else:
            base = torch.zeros(self.num_nodes, device=self._device())
        if self.use_node_domain and self.node_scale is not None:
            base = (
                base
                + torch.abs(self.node_scale.squeeze(-1))
                + torch.abs(self.node_shift.squeeze(-1))
            )
        return base

    def _device(self) -> torch.device:
        for param in self.parameters():
            return param.device
        return torch.device("cpu")


def masked_mae_loss(
    pred: torch.Tensor, target: torch.Tensor, null_val: float = 0.0
) -> torch.Tensor:
    mask = (target != null_val).float()
    mask = mask / torch.clamp(mask.mean(), min=1e-8)
    loss = torch.abs(pred - target) * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss.mean()


def select_horizon_values(
    calibrated: torch.Tensor, horizons: torch.Tensor
) -> torch.Tensor:
    """Select per-row horizon values from ``[K, 1, N, H]`` as ``[K, N]``."""
    values = calibrated[:, 0]
    row_idx = torch.arange(values.shape[0], device=values.device)
    return values[row_idx, :, horizons]


def select_raw_horizon_values(
    raw_pred: torch.Tensor, horizons: torch.Tensor
) -> torch.Tensor:
    values = raw_pred[:, 0]
    row_idx = torch.arange(values.shape[0], device=values.device)
    return values[row_idx, :, horizons]


def count_trainable_params(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)
