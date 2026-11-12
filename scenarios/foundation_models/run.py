"""Run RECAL in the foundation_models setting."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    models = ("timesfm", "moirai", "kairos")
    datasets = ("SD", "GBA", "GLA", "CA")
    p = argparse.ArgumentParser(
        description="RECAL / foundation_models; see README.md for environment and data sources."
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
    p.add_argument("--kairos-source", type=Path)
    p.add_argument("--node-batch", type=int, default=512)
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
    folder = ROOT / "scenarios" / "foundation_models"
    command = [
        sys.executable,
        str(folder / "native.py"),
        "--model",
        a.model,
        "--dataset",
        a.dataset,
        "--data-root",
        str(a.data_root),
        "--checkpoint",
        str(a.checkpoint),
        "--output-dir",
        str(a.output_dir),
        "--device",
        a.device,
        "--seed",
        str(a.seed),
        "--max-steps",
        str(a.max_steps),
    ]
    command += ["--node-batch", str(a.node_batch)]
    if a.kairos_source:
        command += ["--kairos-source", str(a.kairos_source.resolve())]
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
