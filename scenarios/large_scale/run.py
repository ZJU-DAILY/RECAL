"""Run RECAL in the large_scale setting."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    models = ("patchstg", "mage", "sqlinear")
    datasets = ("SD", "GBA", "GLA", "CA")
    p = argparse.ArgumentParser(
        description="RECAL / large_scale; see README.md for environment and data sources."
    )
    p.add_argument("--model", required=True, choices=models)
    p.add_argument("--dataset", required=True, choices=datasets)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument(
        "--checkpoint",
        type=Path,
        help="External file; stage directory for continual; model directory for foundation models",
    )
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Smoke run only; 0 evaluates the full paper stream",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without loading data or models",
    )
    p.add_argument("--mode", choices=("train", "test"), default="test")
    p.add_argument("--batch-size", type=int, default=None)
    a = p.parse_args()
    if a.seed is None:
        a.seed = 2026
    if a.max_steps < 0:
        p.error("--max-steps must be nonnegative")
    if getattr(a, "mode", "test") == "test" and a.checkpoint is None:
        p.error("--checkpoint is required for evaluation")
    a.data_root = a.data_root.resolve()
    a.output_dir = a.output_dir.resolve()
    if a.checkpoint:
        a.checkpoint = a.checkpoint.resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["RECAL_DATA_ROOT"] = str(a.data_root)
    env["RECAL_OUTPUT_DIR"] = str(a.output_dir)
    if a.checkpoint:
        env["RECAL_CHECKPOINT"] = str(a.checkpoint)
    folder = ROOT / "scenarios" / "large_scale"
    checkpoint = (
        a.checkpoint or a.output_dir / f"{a.model}_{a.dataset.lower()}_s{a.seed}.pt"
    )
    command = [
        sys.executable,
        str(folder / "main.py"),
        "--config",
        str(folder / "configs" / f"{a.model}_{a.dataset.lower()}.conf"),
        "--backbone",
        a.model,
        "--cuda",
        a.device.split(":")[-1] if a.device != "cpu" else "-1",
        "--seed",
        str(a.seed),
        "--run_mode",
        a.mode,
        "--batch_size",
        str(1 if a.mode == "test" else a.batch_size or 64),
        "--traffic_file",
        str(a.data_root / a.dataset / f"flow{a.dataset.lower()}.npz"),
        "--meta_file",
        str(a.data_root / a.dataset / f"{a.dataset.lower()}_meta.csv"),
        "--adj_file",
        str(a.data_root / a.dataset / "adj.npy"),
        "--model_file",
        str(checkpoint),
        "--log_file",
        str(a.output_dir / "run.log"),
        "--recal_memory",
        "auto_multi",
        "--recal_max_batches",
        str(a.max_steps),
    ]
    if a.dry_run:
        print(
            json.dumps(
                {"cwd": str(folder), "command": command, "config": None}, indent=2
            )
        )
        return
    if not a.data_root.is_dir():
        p.error(f"Data root does not exist: {a.data_root}")
    if a.checkpoint and (not a.checkpoint.exists()):
        p.error(f"Checkpoint does not exist: {a.checkpoint}")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    raise SystemExit(subprocess.call(command, cwd=folder, env=env))


if __name__ == "__main__":
    main()
