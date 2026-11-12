"""Shared pieces of the long-horizon test-time calibration runs.

The backbones (STID, PatchSTG, BigST; input 96, output 96 or 672 steps) were trained with the
BasicTS/FaST long-horizon pipeline.  This module loads such a checkpoint with the exact BasicTS
config, scaler and test split, exposes the test split as a chronological stream (one window per
tick), and accumulates the BasicTS metrics (masked MAE / RMSE / MAPE, null value 0) over it.
It also holds the selected-horizon update helper for :mod:`recal`: a RECAL assimilator that calibrates only the matured cells of a diagonal.
"""

from __future__ import annotations
import importlib
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
from recal import ForecastDiagonalAssimilator
from recal.calibrator import masked_mae_loss
from recal.controller import FeedbackCell

logger = logging.getLogger(__name__)
NULL_ATOL = 5e-05


@dataclass(frozen=True)
class Backbone:
    """Frozen BasicTS model with the pieces of its config the stream needs."""

    net: nn.Module
    mu: float
    sigma: float
    forward_features: Tuple[int, ...]
    null_val: float
    num_nodes: int


def load_backbone(
    fast_root: str,
    model: str,
    dataset: str,
    input_len: int,
    horizon: int,
    ckpt_path: Optional[str],
    device: torch.device,
) -> Backbone:
    """Build the model of ``backbones/<model>/<dataset>_<P>_<H>.py`` and load ``ckpt_path``.

    Without a checkpoint the weights stay untrained, which is enough to time a horizon that has none.  The BasicTS
    configs resolve dataset files relative to the working directory, so the working directory is switched to
    ``fast_root`` for the rest of the process.
    """
    os.chdir(fast_root)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fast"))
    cfg = importlib.import_module(
        f"backbones.{model}.{dataset}_{input_len}_{horizon}"
    ).CFG
    net = cfg.MODEL.ARCH(**cfg.MODEL.PARAM)
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
        state = {
            k[len("module.") :] if k.startswith("module.") else k: v
            for k, v in state.items()
        }
        net.load_state_dict(state)
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    scaler = cfg.SCALER.TYPE(**cfg.SCALER.PARAM)
    if scaler.mu.numel() != 1:
        raise ValueError(
            "the calibrators assume one global z-score (norm_each_channel=False)"
        )
    return Backbone(
        net=net,
        mu=float(scaler.mu),
        sigma=float(scaler.sigma),
        forward_features=tuple(cfg.MODEL.FORWARD_FEATURES),
        null_val=float(cfg.METRICS.NULL_VAL),
        num_nodes=_num_nodes(cfg.MODEL.PARAM),
    )


def _num_nodes(param: dict) -> int:
    """Node count of a BasicTS model config (``num_nodes`` / ``node_num``, possibly in a nested dict, e.g. BigST)."""
    for key in ("num_nodes", "node_num"):
        if key in param:
            return int(param[key])
    for value in param.values():
        if isinstance(value, dict):
            try:
                return _num_nodes(value)
            except KeyError:
                continue
    raise KeyError("no num_nodes / node_num in the model config")


class TestStream:
    """Test split of ``MyTimeSeries`` as a stream: tick ``i`` serves the ``i``-th test window.

    Window ``i`` reads inputs ``g0 + i + [0, P)`` and targets ``g0 + i + P + [0, H)`` of the
    series, where ``g0 = idx_test[0] - H - P + 1`` (exactly the slicing of ``MyTimeSeries``).
    Inputs are z-scored on channel 0 as the BasicTS scaler does; metric targets go through the
    same transform/inverse round trip as in BasicTS, and calibration labels are the raw values.
    """

    def __init__(
        self,
        dataset: str,
        input_len: int,
        horizon: int,
        bb: Backbone,
        device: torch.device,
    ) -> None:
        series = np.load(f"datasets/{dataset}/his.npz")["data"]
        split = f"datasets/{dataset}/{input_len}_{horizon}"
        idx_train = np.load(f"{split}/idx_train.npy")
        idx_test = np.load(f"{split}/idx_test.npy")
        if not (
            np.all(np.diff(idx_test) == 1) and int(idx_test[-1]) == len(series) - 1
        ):
            raise ValueError(
                "expected contiguous test windows that end with the series"
            )
        self.P, self.H = (int(input_len), int(horizon))
        self.g0 = int(idx_test[0]) - self.H - self.P + 1
        self.train_end = int(idx_train[-1]) + 1
        if self.train_end > self.g0:
            raise ValueError("the training slice overlaps the test stream")
        self.num_ticks = len(idx_test)
        self.series = series
        seg = torch.from_numpy(series[self.g0 :].astype(np.float32)).to(device)
        mu, sigma = (
            torch.tensor(bb.mu, device=device),
            torch.tensor(bb.sigma, device=device),
        )
        norm = seg.clone()
        norm[..., 0] = (seg[..., 0] - mu) / sigma
        self.inputs = norm[..., list(bb.forward_features)].contiguous()
        self.metric_target = (norm[..., 0:1] * sigma + mu).contiguous()
        self.label = seg[..., 0:1].contiguous()
        self.future_stub = torch.zeros(
            1, self.H, seg.shape[1], len(bb.forward_features), device=device
        )
        if seg.shape[0] != self.num_ticks + self.P + self.H - 1:
            raise ValueError("test segment length does not match the number of windows")

    def window(self, i: int) -> torch.Tensor:
        return self.inputs[i : i + self.P].unsqueeze(0)

    def targets(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lo, hi = (i + self.P, i + self.P + self.H)
        return (self.metric_target[lo:hi].unsqueeze(0), self.label[lo:hi].unsqueeze(0))


@torch.no_grad()
def backbone_forecast(bb: Backbone, stream: TestStream, i: int) -> torch.Tensor:
    """Normalized forecast ``[1, H, N, 1]`` of window ``i`` (future inputs are unused by these models).

    Models that return a dict (BigST) put the forecast under ``prediction``, as the BasicTS runner expects.
    """
    out = bb.net(
        history_data=stream.window(i),
        future_data=stream.future_stub,
        batch_seen=None,
        epoch=None,
        train=False,
    )
    return out["prediction"] if isinstance(out, dict) else out


class MaskedMetrics:
    """Streaming BasicTS metrics: sums of masked errors per horizon, in float64."""

    def __init__(self, horizon: int, null_val: float, device: torch.device) -> None:
        self.null_val = float(null_val)
        z = lambda: torch.zeros(horizon, dtype=torch.float64, device=device)
        self.ae, self.se, self.ape, self.cnt, self.ape_cnt = (z(), z(), z(), z(), z())

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """``pred`` and ``target``: ``[1, H, N, 1]`` in original units."""
        mask = ~torch.isclose(
            target, torch.full_like(target, self.null_val), atol=NULL_ATOL, rtol=0.0
        )
        ape_mask = mask & ~torch.isclose(
            target, torch.zeros_like(target), atol=NULL_ATOL
        )
        err = pred - target
        dims = (0, 2, 3)
        self.ae += torch.nan_to_num(err.abs() * mask).sum(dims).double()
        self.se += torch.nan_to_num(err.square() * mask).sum(dims).double()
        self.ape += torch.nan_to_num((err / target).abs() * ape_mask).sum(dims).double()
        self.cnt += mask.sum(dims).double()
        self.ape_cnt += ape_mask.sum(dims).double()

    def result(self) -> Dict[str, object]:
        ae, se, ape = (
            self.ae.cpu().numpy(),
            self.se.cpu().numpy(),
            self.ape.cpu().numpy(),
        )
        cnt, ape_cnt = (self.cnt.cpu().numpy(), self.ape_cnt.cpu().numpy())
        return dict(
            mae=float(ae.sum() / cnt.sum()),
            rmse=float(np.sqrt(se.sum() / cnt.sum())),
            mape=float(ape.sum() / ape_cnt.sum()),
            mae_per_horizon=[round(float(v), 5) for v in ae / np.maximum(cnt, 1.0)],
            rmse_per_horizon=[
                round(float(v), 5) for v in np.sqrt(se / np.maximum(cnt, 1.0))
            ],
            valid_cells=float(cnt.sum()),
        )


class SelectedHorizonAssimilator(ForecastDiagonalAssimilator):
    """Reference RECAL update that calibrates only the matured cells of a diagonal.

    The reference ``_update_cells`` concatenates the whole forecast rows of a diagonal
    (``[K, 1, N, H]``) and calibrates all of them before it selects one horizon per row; at
    ``H = 672`` that costs ``K x N x H`` activations per update.  The calibrator acts on each
    horizon separately, so calibrating the ``K`` selected cells with ``forward_selected`` gives
    the same forward values, and the gradients differ only in the summation order of the node
    terms.  Everything after the forward pass (loss, memory read, verification, write) is the
    reference code unchanged.
    """

    def _update_cells(
        self, cells: List[FeedbackCell], merged_diagonals: int = 1
    ) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start_time = time.perf_counter()
        horizons = torch.tensor(
            [cell.horizon for cell in cells], device=self.device, dtype=torch.long
        )
        raw_sel = torch.stack(
            [cell.raw_pred[0, 0, :, cell.horizon] for cell in cells], dim=0
        ).to(self.device)
        target = torch.stack([cell.target for cell in cells], dim=0).to(self.device)
        self.calibrator.train()
        selected_norm = self.calibrator.forward_selected(raw_sel, horizons)
        pred_orig = selected_norm.unsqueeze(-1) * self.std + self.mean
        depths = None
        fitted = pred_orig
        if self.memory is not None:
            target_times = [cell.target_time for cell in cells]
            offset, lag = (self.calendar.target_offset, self.calendar.input_len)
            self.calendar.check_steps(
                min(target_times) - offset + lag, max(target_times) - offset + lag
            )
            mem_slots = self.calendar.slots_of_targets(
                torch.tensor(target_times, device=self.device)
            )
            with torch.no_grad():
                depths = self.memory.read_depths(mem_slots, horizons)
            if self.memory.config.joint:
                fitted = pred_orig + self.memory.served(depths).unsqueeze(-1)
        loss = masked_mae_loss(fitted, target)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.calibrator.eval()
        if depths is not None:
            with torch.no_grad():
                served, truth = (pred_orig.detach().squeeze(-1), target.squeeze(-1))
                self.memory.verify_depths(
                    truth, served, depths, diagonals=merged_diagonals
                )
                self.memory.write(
                    mem_slots, horizons, truth, served, diagonals=merged_diagonals
                )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.stats.record(
            len(cells),
            (time.perf_counter() - start_time) * 1000.0,
            merged_diagonals=merged_diagonals,
            nodes_updated=int(pred_orig.shape[1]),
            eps_k=0.0,
        )


def median_after_warmup(
    values: Sequence[float], warmup: float = 0.1
) -> Optional[float]:
    """Median of ``values`` after dropping the first ``warmup`` share (protocol timing)."""
    if not values:
        return None
    return float(
        np.median(np.asarray(values[int(len(values) * warmup) :], dtype=np.float64))
    )
