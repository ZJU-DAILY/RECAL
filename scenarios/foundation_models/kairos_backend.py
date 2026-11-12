"""Kairos frozen-buffer median adapter, pinned official source and checkpoint."""

import sys
from pathlib import Path
import numpy as np
import torch

SOURCE = Path(__import__("os").environ["RECAL_KAIROS_SOURCE"])
CHECKPOINT = Path(__import__("os").environ["RECAL_CHECKPOINT"])


class KairosBackend:
    def __init__(self, context=128, horizon=12, node_batch=128, device="cuda"):
        sys.path.insert(0, str(SOURCE))
        from tsfm.model.kairos import AutoModel

        self.checkpoint = CHECKPOINT
        self.name = "kairos-50m"
        self.context = context
        self.horizon = horizon
        self.node_batch = node_batch
        self.device = device
        self.model = (
            AutoModel.from_pretrained(str(CHECKPOINT), local_files_only=True)
            .to(device)
            .float()
            .eval()
        )
        self.frozen_router_count = 0
        for module in self.model.modules():
            if hasattr(module, "update_bias_rate"):
                module.update_bias_rate = 0.0
                self.frozen_router_count += 1
        assert self.frozen_router_count > 0
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.quantiles = self.model.quantiles.detach().cpu().tolist()
        self.median_index = min(
            range(len(self.quantiles)), key=lambda i: abs(self.quantiles[i] - 0.5)
        )
        self.initial_buffers = {
            k: v.detach().cpu().clone() for k, v in self.model.named_buffers()
        }

    @torch.no_grad()
    def forecast(self, history):
        rows = np.ascontiguousarray(history.T, dtype=np.float32)
        assert rows.shape[1] == self.context and np.isfinite(rows).all()
        parts = []
        for start in range(0, len(rows), self.node_batch):
            x = torch.from_numpy(rows[start : start + self.node_batch]).to(self.device)
            out = self.model(
                past_target=x,
                prediction_length=self.horizon,
                generation=True,
                infer_is_positive=True,
                force_flip_invariance=True,
            )
            q = out.prediction_outputs.detach().cpu().numpy()
            assert q.shape == (len(x), len(self.quantiles), self.horizon), q.shape
            parts.append(q[:, self.median_index, :])
        ans = np.concatenate(parts).T.copy()
        assert np.isfinite(ans).all()
        return ans

    def verify_buffers(self):
        assert all(
            torch.equal(v.detach().cpu(), self.initial_buffers[k])
            for k, v in self.model.named_buffers()
        )
