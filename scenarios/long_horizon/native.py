"""RECAL on frozen long-horizon forecasting backbones."""

from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
import platform
import random
import socket
import subprocess
import sys
import time
from collections import deque
from typing import Callable, Dict
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
for path in (HERE, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
from recal import (
    CalendarMemoryConfig,
    ReCalibrator,
    ForecastDiagonalAssimilator,
    build_calendar_index,
    resolve_memory_key,
    select_calendar_key,
    verify_decay_from_halflife,
)
from recal.calendar_memory import REQUEST_KEYS
from recal.calendar_utils import decode_dow, decode_tod
from recal.fused_engine import FusedReCal
from lh_common import (
    Backbone,
    MaskedMetrics,
    SelectedHorizonAssimilator,
    TestStream,
    backbone_forecast,
    load_backbone,
    median_after_warmup,
)

logger = logging.getLogger("recal.long_horizon")
TTA_METHODS = ()
METHODS = ("recal",)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--fast_root",
        required=True,
        help="External root containing datasets/<dataset>/",
    )
    p.add_argument("--model", required=True, choices=("STID", "PatchSTG", "BigST"))
    p.add_argument("--dataset", required=True, choices=("SD", "GBA"))
    p.add_argument("--input_len", type=int, default=96)
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument(
        "--ckpt",
        required=True,
        help="checkpoint, or none for untrained weights (timing only)",
    )
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--max_ticks",
        type=int,
        default=0,
        help="debug: stop after this many ticks (0 = all)",
    )
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--target_offset", type=int, default=1)
    p.add_argument("--memory", default="auto_multi", choices=REQUEST_KEYS)
    p.add_argument("--memory_decay", type=float, default=0.9)
    p.add_argument("--memory_shrink", type=float, default=4.0)
    p.add_argument("--memory_joint", type=int, default=1, choices=(0, 1))
    p.add_argument("--memory_verify", type=int, default=1, choices=(0, 1))
    p.add_argument("--verify_halflife_days", type=float, default=3.6)
    p.add_argument("--memory_null_tol", type=float, default=0.0)
    p.add_argument(
        "--update_path",
        default="selected",
        choices=("selected", "reference", "fused"),
        help="selected: calibrate only the matured cells (same forward values as reference); fused: recal.fused_engine with cell_wise=True (ring buffer + CUDA Graphs)",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def env_info(device: torch.device) -> Dict[str, object]:
    info: Dict[str, object] = dict(
        host=socket.gethostname(),
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,utilization.gpu",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            out = f"nvidia-smi failed: {exc}"
        info["nvidia_smi_at_start"] = out.strip().splitlines()
        info["is_exclusive"] = False
    return info


class TickTimer:
    """Wall time per tick (CUDA-synchronized), excluding metric bookkeeping."""

    def __init__(self, device: torch.device) -> None:
        self.device, self.walls, self._t0 = (device, [], 0.0)

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self) -> None:
        self._sync()
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        self._sync()
        self.walls.append((time.perf_counter() - self._t0) * 1000.0)


def stream_loop(
    stream: TestStream,
    metrics: MaskedMetrics,
    timer: TickTimer,
    num_ticks: int,
    serve: Callable[[int], torch.Tensor],
) -> None:
    """Serve ticks ``0..num_ticks-1`` with ``serve(i) -> [1, H, N, 1]`` and score them."""
    report = max(num_ticks // 10, 1)
    for i in range(num_ticks):
        timer.start()
        pred = serve(i)
        timer.stop()
        metrics.update(pred, stream.targets(i)[0])
        if (i + 1) % report == 0:
            logger.info(
                "tick %d/%d  running MAE %.4f  wall %.1f ms",
                i + 1,
                num_ticks,
                metrics.result()["mae"],
                float(np.mean(timer.walls[-report:])),
            )


def build_calendar(stream: TestStream, args: argparse.Namespace):
    """Calendar of the test stream; key and weekend days are decided on the training slice only."""
    tod, tpd = decode_tod(stream.series[:, 0, 1])
    dow = decode_dow(stream.series[:, 0, 2])
    end = stream.train_end
    auto_key, weekend, score = select_calendar_key(
        stream.series[:end, :, 0],
        tod[:end],
        dow[:end],
        tpd,
        null_tol=args.memory_null_tol,
    )
    key = resolve_memory_key(args.memory, auto_key)
    calendar = build_calendar_index(
        tod,
        dow,
        tpd,
        key,
        first_input_step=stream.g0,
        num_samples=stream.num_ticks,
        seq_len=stream.P,
        horizon=stream.H,
        weekend_dows=weekend,
        slot_ticks=1,
        target_offset=args.target_offset,
    )
    info = dict(
        key=key,
        auto_key=auto_key,
        weekend=[int(d) for d in weekend],
        weekly_score=float(score),
        ticks_per_day=int(tpd),
        key_span=[0, end],
    )
    logger.info("calendar: %s", info)
    return (calendar, info, tpd)


def run_recal(
    bb: Backbone,
    stream: TestStream,
    metrics: MaskedMetrics,
    timer: TickTimer,
    num_ticks: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    device = next(bb.net.parameters()).device
    calendar, cal_info, tpd = (None, dict(key="off"), 96)
    if args.memory != "off":
        calendar, cal_info, tpd = build_calendar(stream, args)
    memory_cfg = CalendarMemoryConfig(
        key=cal_info["key"],
        decay=args.memory_decay,
        shrink=args.memory_shrink,
        joint=bool(args.memory_joint),
        verify=bool(args.memory_verify),
        verify_decay=verify_decay_from_halflife(args.verify_halflife_days, tpd),
        null_tol=args.memory_null_tol,
    )
    torch.manual_seed(args.seed)
    calibrator = (
        ReCalibrator(
            bb.num_nodes,
            stream.H,
            args.rank,
            use_horizon_domain=True,
            use_node_domain=True,
        )
        .to(device)
        .eval()
    )
    if args.update_path == "fused":
        return run_recal_fused(
            bb,
            stream,
            metrics,
            timer,
            num_ticks,
            args,
            calibrator,
            calendar,
            memory_cfg,
            cal_info,
        )
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=args.lr)
    cls = (
        SelectedHorizonAssimilator
        if args.update_path == "selected"
        else ForecastDiagonalAssimilator
    )
    assim = cls(
        calibrator=calibrator,
        optimizer=optimizer,
        mean=bb.mu,
        std=max(bb.sigma, 1e-08),
        device=device,
        target_offset=args.target_offset,
        update_interval=1,
        maintenance_budget=0,
        feedback_mode="legacy",
        execution_backend="reference",
        calendar=calendar,
        memory=memory_cfg,
    )

    def serve(i: int) -> torch.Tensor:
        assim.maintain(i)
        y_hat = backbone_forecast(bb, stream, i).permute(0, 3, 2, 1)
        with torch.no_grad():
            served = calibrator(y_hat).permute(0, 3, 2, 1) * bb.sigma + bb.mu
            recalled = assim.memory_correction(i, 1)
            if recalled is not None:
                served = served + recalled.unsqueeze(-1)
        assim.enqueue(i, y_hat, stream.targets(i)[1])
        return served

    stream_loop(stream, metrics, timer, num_ticks, serve)
    st = assim.stats
    return dict(
        calendar=cal_info,
        memory=assim.memory_stats,
        update_path=args.update_path,
        applied_updates=st.applied,
        avg_cells=st.avg_cells,
        avg_adapt_ms=st.avg_adapt_ms,
        max_working_set=assim.max_working_set,
        rank=args.rank,
        lr=args.lr,
        params=int(sum((p.numel() for p in calibrator.parameters()))),
    )


def run_recal_fused(
    bb: Backbone,
    stream: TestStream,
    metrics: MaskedMetrics,
    timer: TickTimer,
    num_ticks: int,
    args: argparse.Namespace,
    calibrator,
    calendar,
    memory_cfg,
    cal_info: Dict[str, object],
) -> Dict[str, object]:
    """RECAL through the fused engine: maintain, serve and enqueue in one ``step`` per tick."""
    device = next(bb.net.parameters()).device
    engine = FusedReCal(
        calibrator,
        num_nodes=bb.num_nodes,
        pred_len=stream.H,
        mean=bb.mu,
        std=max(bb.sigma, 1e-08),
        batch_size=1,
        device=device,
        lr=args.lr,
        target_offset=args.target_offset,
        update_interval=1,
        calendar=calendar,
        memory=memory_cfg if memory_cfg.enabled else None,
        cell_wise=True,
    )

    def serve(i: int) -> torch.Tensor:
        y_hat = backbone_forecast(bb, stream, i).permute(0, 3, 2, 1)
        return engine.step(i, y_hat, stream.targets(i)[1]).unsqueeze(-1)

    stream_loop(stream, metrics, timer, num_ticks, serve)
    return dict(
        calendar=cal_info,
        update_path="fused",
        engine=engine.summary(),
        rank=args.rank,
        lr=args.lr,
        params=int(sum((p.numel() for p in calibrator.parameters()))),
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = None if args.ckpt == "none" else os.path.abspath(args.ckpt)
    out = os.path.abspath(args.out)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    meta = dict(env=env_info(device), started=started, argv=sys.argv[1:])
    bb = load_backbone(
        os.path.abspath(args.fast_root),
        args.model,
        args.dataset,
        args.input_len,
        args.horizon,
        ckpt,
        device,
    )
    stream = TestStream(args.dataset, args.input_len, args.horizon, bb, device)
    num_ticks = (
        min(stream.num_ticks, args.max_ticks) if args.max_ticks else stream.num_ticks
    )
    logger.info(
        "%s %s %d->%d %s: %d ticks, N=%d, mu=%.4f sigma=%.4f",
        args.model,
        args.dataset,
        args.input_len,
        args.horizon,
        args.method,
        num_ticks,
        bb.num_nodes,
        bb.mu,
        bb.sigma,
    )
    metrics = MaskedMetrics(stream.H, bb.null_val, device)
    timer = TickTimer(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    runner = run_recal
    extra = runner(bb, stream, metrics, timer, num_ticks, args)
    result = dict(
        model=args.model,
        dataset=args.dataset,
        input_len=args.input_len,
        horizon=args.horizon,
        method=args.method,
        num_ticks=num_ticks,
        total_ticks=stream.num_ticks,
        num_nodes=bb.num_nodes,
        ckpt=ckpt,
        ckpt_md5=file_md5(ckpt) if ckpt else None,
        seed=args.seed,
        **metrics.result(),
        method_info=extra,
    )
    result["timing"] = dict(
        wall_ms_median_after_10pct=median_after_warmup(timer.walls),
        wall_ms_mean=float(np.mean(timer.walls)),
        total_s=float(np.sum(timer.walls)) / 1000.0,
        peak_mem_mb=(
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else 0.0
        ),
    )
    result["meta"] = dict(meta, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(
        "MAE %.4f RMSE %.4f MAPE %.4f | median %.1f ms/tick | -> %s",
        result["mae"],
        result["rmse"],
        result["mape"],
        result["timing"]["wall_ms_median_after_10pct"] or -1.0,
        out,
    )


if __name__ == "__main__":
    main()
