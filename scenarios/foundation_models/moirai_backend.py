"""Frozen Moirai-2 median adapter following ORCA's Moirai2Forecast.predict path."""

from pathlib import Path
import numpy as np
import torch

CHECKPOINT = Path(__import__("os").environ["RECAL_CHECKPOINT"])


class MoiraiBackend:
    def __init__(self, context=128, horizon=12, node_batch=128):
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

        self.checkpoint = CHECKPOINT
        self.name = "moirai-2"
        self.context = context
        self.horizon = horizon
        self.node_batch = node_batch
        self.model = (
            Moirai2Forecast(
                module=Moirai2Module.from_pretrained(
                    str(CHECKPOINT), local_files_only=True
                ),
                prediction_length=horizon,
                context_length=context,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            .to(__import__("os").environ.get("RECAL_DEVICE", "cuda"))
            .float()
            .eval()
        )
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.quantiles = list(self.model.module.quantile_levels)
        self.median_index = next(
            i for i, q in enumerate(self.quantiles) if abs(float(q) - 0.5) < 1e-8
        )

    @torch.no_grad()
    def forecast(self, history):
        rows = np.ascontiguousarray(history.T, dtype=np.float32)
        assert rows.shape[1] == self.context and np.isfinite(rows).all()
        parts = []
        for start in range(0, len(rows), self.node_batch):
            q = np.asarray(
                self.model.predict(list(rows[start : start + self.node_batch])),
                dtype=np.float32,
            )
            if q.ndim == 4 and q.shape[-1] == 1:
                q = q[..., 0]
            assert q.shape == (
                min(self.node_batch, len(rows) - start),
                len(self.quantiles),
                self.horizon,
            ), q.shape
            parts.append(q[:, self.median_index, :])
        ans = np.concatenate(parts).T.copy()
        assert ans.shape == (self.horizon, len(rows)) and np.isfinite(ans).all()
        return ans
