"""Create paper few-shot index prefixes; preserve the full-training scaler."""

import argparse
from pathlib import Path
import shutil
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument(
        "--dataset", choices=("PEMS03", "PEMS04", "PEMS07", "PEMS08"), required=True
    )
    p.add_argument("--ratio", type=int, choices=(5, 10, 20), required=True)
    a = p.parse_args()
    year = {"PEMS03": "2018", "PEMS04": "2018", "PEMS07": "2017", "PEMS08": "2016"}[
        a.dataset
    ]
    base = a.data_root.resolve() / a.dataset.lower()
    source = base / year
    target = base / f"{year}-fewshot{a.ratio}"
    idx = np.load(source / "idx_train.npy")
    if not np.all(np.diff(idx) == 1):
        raise ValueError("Training indices must be chronological and contiguous")
    count = int(len(idx) * a.ratio / 100)
    target.mkdir(exist_ok=False)
    try:
        (target / "his.npz").symlink_to(source / "his.npz")
    except OSError:
        shutil.copyfile(source / "his.npz", target / "his.npz")
    np.save(target / "idx_train.npy", idx[:count])
    for split in ("val", "test"):
        shutil.copyfile(source / f"idx_{split}.npy", target / f"idx_{split}.npy")
    print(
        f"{target}: {count}/{len(idx)} training windows; scaler and evaluation indices preserved"
    )


if __name__ == "__main__":
    main()
