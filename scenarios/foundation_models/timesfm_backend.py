"""Frozen, independent-node TSFM adapters; all inputs/outputs use original units."""

from pathlib import Path
import numpy as np
import torch

CHECKPOINT = Path(__import__("os").environ["RECAL_CHECKPOINT"])


class Backend:

    def __init__(self, name, context=128, horizon=12, node_batch=128):
        self.name, self.context, self.horizon, self.node_batch = (
            name,
            context,
            horizon,
            node_batch,
        )
        import timesfm

        self.pipeline = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            CHECKPOINT, local_files_only=True, torch_compile=False
        )
        self.pipeline.compile(
            timesfm.ForecastConfig(
                max_context=context,
                max_horizon=horizon,
                normalize_inputs=True,
                per_core_batch_size=node_batch,
                use_continuous_quantile_head=False,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=False,
            )
        )
        self.model = self.pipeline.model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forecast(self, history):
        rows = np.ascontiguousarray(history.T, dtype=np.float32)
        point, _ = self.pipeline.forecast(self.horizon, list(rows))
        answer = np.asarray(point, dtype=np.float32).T
        assert answer.shape == (self.horizon, rows.shape[0]), answer.shape
        if not np.isfinite(answer).all():
            raise ValueError("Non-finite TSFM prediction")
        return answer
