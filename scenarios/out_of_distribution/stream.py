"""RECAL streaming adapter for the frozen REACT and STONE backbones."""

import os
import time
from collections import deque
from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
from recal import (
    ReCalibrator,
    ForecastDiagonalAssimilator,
    LEGACY_FEEDBACK,
    count_trainable_params,
)
from recal.fused_engine import FusedReCal
from adapters.react import compute_metrics, mask_value, set_seed
from scaler import StandardScaler
from metrics import masked_mae


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _scalar(value) -> float:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value).reshape(-1)[0])


def load_test_stream(
    data_path: str, args, node_indices: np.ndarray, stream_multiple: int = 64
):
    """Test series of one scenario restricted to the test nodes, its sample indices and its scaler.

    Same source as ``load_dataset`` (test year ``checkyears`` when tood, else the training year) but without loading the
    train / val loaders the streaming evaluation does not use.  ``stream_multiple`` drops the tail that does not fill a
    batch of REACT's evaluation (bs=64), so the base row is comparable with the published numbers.
    """
    train_ptr = np.load(os.path.join(data_path, str(args.years), "his.npz"))
    if int(args.tood):
        year = str(args.checkyears)
        ptr = np.load(os.path.join(data_path, year, "his.npz"))
    else:
        year, ptr = (str(args.years), train_ptr)
    scaler = StandardScaler(mean=ptr["mean"], std=ptr["std"])
    series = np.ascontiguousarray(
        ptr["data"][..., : int(args.input_dim)][:, node_indices, :], dtype=np.float32
    )
    idx = np.load(os.path.join(data_path, year, "idx_test.npy"))
    if stream_multiple and stream_multiple > 1:
        idx = idx[: len(idx) // stream_multiple * stream_multiple]
    return (series, idx, scaler, year)


class OODStream:
    """Shared OOD evaluation stream for STONE and REACT."""

    def __init__(
        self,
        model: nn.Module,
        series: np.ndarray,
        idx: np.ndarray,
        scaler: StandardScaler,
        test_nodes: np.ndarray,
        node_unseen_num: int,
        seq_len: int,
        horizon: int,
        device: torch.device,
        batch_size: int = 1,
    ):
        self.model = model.eval()
        self.series, self.idx = (series, idx)
        self.scaler = scaler
        self.mean, self.std = (_scalar(scaler.mean), max(_scalar(scaler.std), 1e-08))
        self.device = torch.device(device)
        self.test_nodes = torch.as_tensor(
            test_nodes, dtype=torch.long, device=self.device
        )
        self.node_num = int(len(test_nodes))
        self.unseen = int(node_unseen_num)
        self.horizon = int(horizon)
        self.batch_size = int(batch_size)
        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, horizon + 1, 1)

    def batches(self, max_batches: int = 0):
        """Yield ``(start, x, y)``: x ``[B, T, N, C]`` and normalised labels y ``[B, H, N, 1]`` on the device."""
        total = len(self.idx)
        count = 0
        for start in range(0, total, self.batch_size):
            if max_batches and count >= max_batches:
                break
            end = min(total, start + self.batch_size)
            rows = self.idx[start:end, None]
            x = torch.from_numpy(self.series[rows + self.x_offsets]).to(self.device)
            y = torch.from_numpy(self.series[rows + self.y_offsets][..., :1]).to(
                self.device
            )
            yield (start, x, y)
            count += 1

    def forecast(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen backbone forecast ``[B, H, N, 1]`` in normalised units."""
        out = self.model(x, node_indices=self.test_nodes)
        return out["pred"] if isinstance(out, dict) else out

    def _inverse(self, t: torch.Tensor) -> torch.Tensor:
        return self.scaler.inverse_transform(t)

    def run_recal(
        self,
        rank: int = 4,
        lr: float = 0.0001,
        target_offset: int = 1,
        update_interval: int = 1,
        calendar=None,
        memory=None,
        seed: int = 3407,
        max_batches: int = 0,
        label: str = "RECAL",
        backend: str = "reference",
    ) -> Dict:
        """RECAL on the stream; ``backend='fused'`` = CUDA-graph execution of the same update (recal/fused_engine.py)."""
        set_seed(seed)
        calibrator = ReCalibrator(
            self.node_num,
            self.horizon,
            rank,
            use_horizon_domain=True,
            use_node_domain=True,
        ).to(self.device)
        calibrator.eval()
        fused = backend == "fused"
        assimilator, engine = (None, None)
        if not fused:
            optimizer = torch.optim.Adam(calibrator.parameters(), lr=lr)
            assimilator = ForecastDiagonalAssimilator(
                calibrator=calibrator,
                optimizer=optimizer,
                mean=self.mean,
                std=self.std,
                device=self.device,
                target_offset=target_offset,
                update_interval=update_interval,
                maintenance_budget=0,
                execution_backend="reference",
                feedback_mode=LEGACY_FEEDBACK,
                allowed_lateness=0,
                calendar=calendar,
                memory=memory,
            )
        preds, labels, infer, wall = ([], [], [], [])
        self._reset_memory_stats()
        for start, x, y in self.batches_timed(wall, max_batches):
            target = self._inverse(y)
            if fused:
                _sync(self.device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    raw = self.forecast(x).permute(0, 3, 2, 1)
                if engine is None:
                    engine = FusedReCal(
                        calibrator,
                        num_nodes=self.node_num,
                        pred_len=self.horizon,
                        mean=self.mean,
                        std=self.std,
                        batch_size=raw.shape[0],
                        device=self.device,
                        lr=lr,
                        target_offset=target_offset,
                        update_interval=update_interval,
                        calendar=calendar,
                        memory=memory,
                    )
                cal, recalled = engine.step(start, raw, target, parts=True)
                pred = self._inverse(cal.permute(0, 3, 2, 1))
                if recalled is not None:
                    pred = pred + recalled.unsqueeze(-1).to(pred.device)
                _sync(self.device)
                infer.append((time.perf_counter() - t0) * 1000.0)
            else:
                assimilator.maintain(start)
                _sync(self.device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    raw = self.forecast(x).permute(0, 3, 2, 1)
                    calibrator.eval()
                    yc = calibrator(raw).permute(0, 3, 2, 1)
                    recalled = assimilator.memory_correction(start, yc.shape[0])
                _sync(self.device)
                infer.append((time.perf_counter() - t0) * 1000.0)
                pred = self._inverse(yc)
                if recalled is not None:
                    pred = pred + recalled.unsqueeze(-1).to(pred.device)
            preds.append(pred.squeeze(-1).cpu())
            labels.append(target.squeeze(-1).cpu())
            if not fused:
                assimilator.enqueue(start, raw, target)
        summary = self._summary(
            label,
            preds,
            labels,
            infer,
            [],
            wall,
            params=count_trainable_params(calibrator),
        )
        if fused:
            summary["efficiency"]["adapt_ms"] = 0.0
            stats = engine.summary() if engine is not None else {}
            summary["update_stats"] = (
                "fused: " + str({k: v for k, v in stats.items() if k != "memory"})[:300]
            )
            floats = (
                stats.get("memory_state_floats") if isinstance(stats, dict) else None
            )
            if floats:
                summary["memory_state_floats"] = int(floats)
        else:
            summary["efficiency"]["adapt_ms"] = float(assimilator.stats.avg_adapt_ms)
            summary["update_stats"] = (
                f"applied={assimilator.stats.applied}, eligible={assimilator.stats.eligible}"
            )
            summary["calendar_memory"] = str(assimilator.memory_stats)[:700]
            if isinstance(
                assimilator.memory_stats, dict
            ) and assimilator.memory_stats.get("state_floats"):
                summary["memory_state_floats"] = int(
                    assimilator.memory_stats["state_floats"]
                )
        return summary

    def batches_timed(self, wall: List, max_batches: int = 0):
        """``batches`` that appends ``(ms, samples)`` of the whole loop body (data prep, serving, adaptation)."""
        _sync(self.device)
        tick = time.perf_counter()
        for start, x, y in self.batches(max_batches):
            yield (start, x, y)
            _sync(self.device)
            now = time.perf_counter()
            wall.append(((now - tick) * 1000.0, x.shape[0]))
            tick = now

    def _reset_memory_stats(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _summary(
        self, label: str, preds, labels, infer, adapt, wall, params: int
    ) -> Dict:
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        metrics = compute_metrics(preds, labels, self.unseen, self.horizon)
        tail = wall[max(1, len(wall) // 10) :] or wall
        wall_ms = [ms for ms, _ in tail]
        infer_sec = sum(infer) / 1000.0
        samples = int(preds.shape[0])
        return {
            "method_label": label,
            "metrics": metrics,
            "samples": samples,
            "efficiency": {
                "params": int(params),
                "peak_mem_mb": (
                    torch.cuda.max_memory_allocated(self.device) / 1024**2
                    if self.device.type == "cuda"
                    else 0.0
                ),
                "samples_per_sec": samples / infer_sec if infer_sec > 0 else 0.0,
                "infer_ms": float(np.mean(infer)) if infer else 0.0,
                "adapt_ms": float(np.mean(adapt)) if adapt else 0.0,
                "wall_ms_median": float(np.median(wall_ms)),
                "wall_samples_per_sec": sum((n for _, n in tail))
                / (sum(wall_ms) / 1000.0),
                "total_s": sum((ms for ms, _ in wall)) / 1000.0,
            },
        }
