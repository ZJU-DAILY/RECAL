"""Run RECAL in the continual setting."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    models = ("eac", "stkec")
    datasets = ("PEMS", "AIR", "ENERGY-Wind")
    p = argparse.ArgumentParser(
        description="RECAL / continual; see README.md for environment and data sources."
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
    p.add_argument(
        "--prepared-dir",
        type=Path,
        help="Explicit strict-causal FastData directory; default <data-root>/<dataset>/strict_causal",
    )
    a = p.parse_args()
    if a.seed is None:
        a.seed = 43 if a.dataset == "AIR" and a.model == "eac" else 42
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
    config = None
    folder = ROOT / "scenarios" / "continual"
    config = json.loads(
        (folder / "configs" / f"{a.model}_{a.dataset.lower()}.json").read_text()
    )
    if a.seed in (2024, 2025):
        config = json.loads(
            (
                folder / "configs" / f"{a.model}_{a.dataset.lower()}_s{a.seed}.json"
            ).read_text()
        )
    config.update(
        raw_data_path=str(a.data_root / a.dataset / "RawData"),
        save_data_path=str(
            a.prepared_dir.resolve()
            if a.prepared_dir
            else a.data_root / a.dataset / "strict_causal"
        ),
        graph_path=str(a.data_root / a.dataset / "graph"),
        model_path=str(a.output_dir),
        logname="recal",
        seed=a.seed,
    )
    env["RECAL_STAGE_CHECKPOINT_ROOT"] = str(a.checkpoint)
    command = [
        sys.executable,
        str(folder / "native.py"),
        "--conf",
        str(a.output_dir / "config.json"),
        "--gpuid",
        a.device.split(":")[-1] if a.device != "cpu" else "-1",
        "--seed",
        str(a.seed),
        "--method",
        a.model.upper(),
        "--recal_memory",
        "auto_multi",
        "--recal_memory_ticks_per_day",
        str({"PEMS": 288, "AIR": 24, "ENERGY-Wind": 144}[a.dataset]),
        "--recal_max_batches",
        str(a.max_steps),
    ]
    if a.dry_run:
        print(
            json.dumps(
                {"cwd": str(folder), "command": command, "config": config}, indent=2
            )
        )
        return
    if not a.data_root.is_dir():
        p.error(f"Data root does not exist: {a.data_root}")
    if a.checkpoint and (not a.checkpoint.exists()):
        p.error(f"Checkpoint does not exist: {a.checkpoint}")
    if not a.checkpoint.is_dir():
        p.error(
            "Continual --checkpoint must be a directory containing one subdirectory per stage"
        )
    for year in range(config["begin_year"], config["end_year"] + 1):
        if not list((a.checkpoint / str(year)).glob("*.pkl")):
            p.error(f"Missing stage checkpoint: {a.checkpoint / str(year)}")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if config:
        (a.output_dir / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
    raise SystemExit(subprocess.call(command, cwd=folder, env=env))


if __name__ == "__main__":
    main()
