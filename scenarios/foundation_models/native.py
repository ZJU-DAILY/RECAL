"""RECAL with the fixed foundation-model configurations selected for the paper."""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from recal import (
    ReCalibrator,
    ForecastDiagonalAssimilator,
    per_step_series,
    select_calendar_key,
    resolve_memory_key,
    build_calendar_index,
    memory_config_from_options,
)
from metrics import Metrics

# (learning rate, shrinkage, verification half-life in days), frozen before testing.
PRESETS = {
    "timesfm": {d: (1e-4, 4, 3.6) for d in ("SD", "GBA", "GLA", "CA")},
    "moirai": {
        "SD": (2e-4, 4, 0.5),
        "GBA": (3e-4, 16, 3.6),
        "GLA": (2e-4, 4, 1.0),
        "CA": (1e-4, 4, 3.6),
    },
    "kairos": {
        "SD": (2e-4, 4, 0.5),
        "GBA": (2e-4, 4, 1.0),
        "GLA": (2e-4, 4, 1.0),
        "CA": (2e-4, 4, 0.5),
    },
}


class Stream:
    def __init__(self, path, count=1024):
        path = path if path.exists() else path.with_suffix(".npy")
        if path.suffix == ".npy":
            self.values = np.load(path, mmap_mode="r")[..., 0].astype(np.float32)
        else:
            with np.load(path) as z:
                self.values = z["data"][..., 0].astype(np.float32)
        t, n = self.values.shape
        train = round(0.6 * t)
        test = round(0.2 * t)
        self.train_end = train
        self.test_start = t - test
        # Match the native LargeST P=12,Q=12 input-window-weighted scaler.
        weights = np.convolve(np.ones(train - 23), np.ones(12))
        sums = np.empty(len(weights))
        squares = np.empty(len(weights))
        for lo in range(0, len(weights), 1024):
            block = self.values[lo : min(lo + 1024, len(weights))].astype(np.float64)
            sums[lo : lo + len(block)] = block.sum(1)
            squares[lo : lo + len(block)] = (block * block).sum(1)
        denominator = weights.sum() * n
        self.mean = float(weights @ sums / denominator)
        self.std = float(np.sqrt(weights @ squares / denominator - self.mean**2))
        self.values[np.abs(self.values) <= 0.01] = 0
        self.idx = np.arange(self.test_start + 11, t - 12)[:count]
        if len(self.idx) != count:
            raise ValueError("Not enough contiguous test origins")
        self.nodes = n
        self.ticks = 96
        steps = np.arange(t)
        self.tod = steps % 96
        self.dow = (steps // 96) % 7

    def history(self, t):
        at = int(self.idx[t])
        return self.values[at - 127 : at + 1]

    def labels(self, t):
        at = int(self.idx[t])
        return self.values[at + 1 : at + 13]

    def calendar(self, shrink, half_life):
        first = int(self.idx[0]) - 127
        stop = min(self.train_end, first)
        series = per_step_series(self.values[:stop], 0.01)
        auto, weekend, score = select_calendar_key(
            series, self.tod[:stop], self.dow[:stop], self.ticks
        )
        key = resolve_memory_key("auto_multi", auto)
        calendar = build_calendar_index(
            self.tod,
            self.dow,
            self.ticks,
            key,
            first,
            len(self.idx),
            128,
            12,
            weekend,
            target_offset=1,
        )
        memory = memory_config_from_options(
            key,
            self.ticks,
            decay=0.9,
            shrink=shrink,
            joint=True,
            verify=True,
            verify_halflife_days=half_life,
            null_tol=0.01,
        )
        return (
            calendar,
            memory,
            dict(
                key=key,
                weekend=list(weekend),
                selection_span=[0, stop],
                weekly_score=float(score),
            ),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=tuple(PRESETS), required=True)
    p.add_argument("--dataset", choices=("SD", "GBA", "GLA", "CA"), required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--kairos-source")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--node-batch", type=int, default=512)
    a = p.parse_args()
    if a.device == "cpu":
        p.error(
            "Paper foundation backends require CUDA; use separate model environments in README."
        )
    torch.cuda.set_device(torch.device(a.device))
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if a.model != "timesfm" or a.dataset in ("SD", "GLA"):
        torch.backends.cudnn.allow_tf32 = False
    os.environ["RECAL_CHECKPOINT"] = a.checkpoint
    os.environ["RECAL_DEVICE"] = a.device
    if a.model == "timesfm":
        from timesfm_backend import Backend

        backend = Backend("timesfm-2.5", node_batch=a.node_batch)
    elif a.model == "moirai":
        from moirai_backend import MoiraiBackend

        backend = MoiraiBackend(node_batch=a.node_batch)
    else:
        if not a.kairos_source:
            p.error("--kairos-source is required")
        os.environ["RECAL_KAIROS_SOURCE"] = a.kairos_source
        from kairos_backend import KairosBackend

        backend = KairosBackend(node_batch=a.node_batch, device=a.device)
    count = min(a.max_steps, 1024) if a.max_steps else 1024
    stream = Stream(a.data_root / a.dataset / f"flow{a.dataset.lower()}.npz", count)
    lr, shrink, half_life = PRESETS[a.model][a.dataset]
    # Preserve the original capture-then-replay order and calibrator RNG reset.
    raw = np.stack([backend.forecast(stream.history(i)) for i in range(count)])
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    cal = ReCalibrator(stream.nodes, 12, 4).to(a.device)
    opt = torch.optim.Adam(cal.parameters(), lr=lr)
    calendar, memory, meta = stream.calendar(shrink, half_life)
    assim = ForecastDiagonalAssimilator(
        cal,
        opt,
        stream.mean,
        stream.std,
        torch.device(a.device),
        target_offset=1,
        update_interval=1,
        maintenance_budget=0,
        feedback_mode="legacy",
        execution_backend="reference",
        calendar=calendar,
        memory=memory,
    )
    metrics = Metrics(12, stream.nodes)
    for t in range(count):
        assim.maintain(t)
        with torch.no_grad():
            cal.eval()
            tensor = torch.as_tensor(
                (raw[t] - stream.mean) / stream.std,
                device=a.device,
                dtype=torch.float32,
            ).T[None, None]
            pred = cal(tensor).squeeze(0).squeeze(0).T * stream.std + stream.mean
            recalled = assim.memory_correction(t, 1)
            if recalled is not None:
                pred += recalled.squeeze(0)
            metrics.add(pred.cpu().numpy(), stream.labels(t))
        target = torch.as_tensor(
            stream.labels(t)[..., None], device=a.device
        ).unsqueeze(0)
        assim.enqueue(t, tensor, target)
    if hasattr(backend, "verify_buffers"):
        backend.verify_buffers()
    result = dict(
        model=a.model,
        dataset=a.dataset,
        seed=a.seed,
        context=128,
        horizon=12,
        origins=count,
        params=dict(
            rank=4, lr=lr, decay=0.9, shrink=shrink, verify_halflife_days=half_life
        ),
        calendar=meta,
        metrics=metrics.result(),
        node_inference_batch=a.node_batch,
        is_exclusive=False,
        efficiency_usable=False,
    )
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
