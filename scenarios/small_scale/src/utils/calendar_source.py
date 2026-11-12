"""Calendar source of the small-scale (full-shot and few-shot) test streams.

RECAL's calendar memory needs the time of day / day of week of every stream
step.  ``his.npz`` stores them in channels 1 and 2 (fractions or integers,
decoded convention-free); the calendar key and the weekend days are decided on
data that precede the test stream only.
"""

import os

import numpy as np

from recal import decode_dow, decode_tod, per_step_series

# Key selection uses the training span; shorter spans (few-shot) use all pre-test steps.
# Four weeks keep two of every weekday in each half of the cross-fitted weekly test.
MIN_SELECT_DAYS = 28


def dataset_folder(args) -> str:
    """Mirror of the folder logic in ``src/utils/dataloader.py::load_dataset``."""
    folder_name = (
        args.years + f"-fewshot{args.fewshot}" if args.fewshot != 100 else args.years
    )
    if args.seq_len != 12 or args.horizon != 12:
        folder_name = folder_name + "-long"
    return folder_name


def calendar_memory_kwargs(args, data_path) -> dict:
    """Keyword arguments for ``BaseEngine.evaluate_with_recal`` ({} when the memory is off)."""
    requested = getattr(args, "recal_memory", "off")
    if requested == "off":
        return {}
    folder = os.path.join(data_path, dataset_folder(args))
    ptr = np.load(os.path.join(folder, "his.npz"))
    data = ptr["data"]
    if data.ndim != 3 or data.shape[-1] < 3:
        raise ValueError(
            "calendar memory needs time-of-day / day-of-week channels in his.npz"
        )
    tod, ticks_per_day = decode_tod(data[:, 0, 1])
    dow = decode_dow(data[:, 0, 2])
    mean = float(np.asarray(ptr["mean"]).reshape(-1)[0])
    std = float(np.asarray(ptr["std"]).reshape(-1)[0])
    idx_train = np.load(os.path.join(folder, "idx_train.npy"))
    idx_test = np.load(os.path.join(folder, "idx_test.npy"))
    if not np.all(np.diff(idx_test) == 1):
        raise ValueError("calendar memory needs a contiguous chronological test stream")

    first_test_input = int(idx_test[0]) - args.seq_len + 1
    train_lo = int(idx_train[0]) - args.seq_len + 1
    train_hi = min(int(idx_train[-1]) + args.horizon + 1, first_test_input)
    if train_hi - train_lo < MIN_SELECT_DAYS * ticks_per_day:
        select_span = slice(0, first_test_input)
    else:
        select_span = slice(train_lo, train_hi)

    # Pre-test values in original units, reduced to one node-averaged series (missing excluded).
    series = per_step_series(
        data[:first_test_input, :, 0] * std + mean, args.recal_memory_null_tol
    )
    weekend = getattr(args, "recal_memory_weekend", "")
    return dict(
        recal_memory=requested,
        recal_memory_source=dict(
            values=series,
            tod=tod,
            dow=dow,
            ticks_per_day=ticks_per_day,
            select_span=select_span,
            first_input_step=first_test_input,
            num_samples=len(idx_test),
        ),
        recal_memory_options=dict(
            decay=args.recal_memory_decay,
            shrink=args.recal_memory_shrink,
            root_shrink=args.recal_memory_root_shrink,
            joint=bool(args.recal_memory_joint),
            verify=bool(args.recal_memory_verify),
            verify_halflife_days=args.recal_memory_verify_halflife_days,
            null_tol=args.recal_memory_null_tol,
        ),
        recal_memory_slot_ticks=args.recal_memory_slot_ticks,
        recal_memory_weekend=(
            tuple(int(d) for d in weekend.split(",")) if weekend else None
        ),
    )
