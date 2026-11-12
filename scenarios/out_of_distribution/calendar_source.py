"""Calendar source of a REACT scenario for RECAL's calendar memory (C3).

Same construction as ``src/utils/calendar_source.py`` (STONE: train on year Y, test on Y + 1), generalised to REACT's
scenarios, where the checkpoint is trained on one year (2017) and the stream is the test split of any year 2017-2021.
The calendar key and the weekend days are decided on the TRAINING year with the nodes seen during training only; the index
is built for the test stream, whose node axis is REACT's ``[seen, unseen]`` test-node order.
"""

import os
from typing import Dict, Optional

import numpy as np

from recal import (
    build_calendar_index,
    decode_dow,
    decode_tod,
    memory_config_from_options,
    per_step_series,
    resolve_memory_key,
    select_calendar_key,
)


def _calendar_channels(path: str, ticks_per_day: Optional[int] = None):
    ptr = np.load(path)
    data = ptr["data"]
    if data.ndim != 3 or data.shape[-1] < 3:
        raise ValueError(
            f"calendar memory needs time-of-day / day-of-week channels in {path}"
        )
    tod, ticks_per_day = decode_tod(data[:, 0, 1], ticks_per_day=ticks_per_day)
    return (
        data,
        tod,
        decode_dow(data[:, 0, 2]),
        ticks_per_day,
        float(np.asarray(ptr["mean"]).reshape(-1)[0]),
        float(np.asarray(ptr["std"]).reshape(-1)[0]),
    )


def calendar_memory_kwargs(
    data_path: str,
    train_year: str,
    test_year: str,
    train_nodes: np.ndarray,
    num_samples: int,
    seq_len: int,
    horizon: int,
    requested: str,
    target_offset: int,
    logger,
    decay: float = 0.9,
    shrink: float = 4.0,
    root_shrink: Optional[float] = None,
    joint: bool = True,
    verify: bool = True,
    verify_halflife_days: float = 3.6,
    null_tol: float = 0.01,
    slot_ticks: int = 1,
    weekend_override: Optional[tuple] = None,
) -> Dict:
    """``calendar`` / ``memory`` keyword arguments for ``ForecastDiagonalAssimilator`` ({} when ``requested == 'off'``)."""
    if requested == "off":
        return {}
    train_data, tod_tr, dow_tr, ticks_per_day, mean, std = _calendar_channels(
        os.path.join(data_path, train_year, "his.npz")
    )
    idx_train = np.load(os.path.join(data_path, train_year, "idx_train.npy"))
    span = slice(int(idx_train[0]) - seq_len + 1, int(idx_train[-1]) + horizon + 1)
    values = train_data[span][:, np.asarray(train_nodes), 0]
    series = per_step_series(values * std + mean, null_tol)
    auto_key, weekend, score = select_calendar_key(
        series,
        tod_tr[span],
        dow_tr[span],
        ticks_per_day,
        weekend_override=weekend_override,
    )
    key = resolve_memory_key(requested, auto_key)

    if test_year == train_year:
        tod_te, dow_te = tod_tr, dow_tr
    else:
        _, tod_te, dow_te, _, _, _ = _calendar_channels(
            os.path.join(data_path, test_year, "his.npz"), ticks_per_day
        )
    idx_test = np.load(os.path.join(data_path, test_year, "idx_test.npy"))
    if not np.all(np.diff(idx_test) == 1):
        raise ValueError("calendar memory needs a contiguous chronological test stream")
    calendar = build_calendar_index(
        tod_te,
        dow_te,
        ticks_per_day,
        key,
        first_input_step=int(idx_test[0]) - seq_len + 1,
        num_samples=int(num_samples),
        seq_len=seq_len,
        horizon=horizon,
        weekend_dows=weekend,
        slot_ticks=slot_ticks,
        target_offset=target_offset,
    )
    memory = memory_config_from_options(
        key,
        ticks_per_day,
        decay=decay,
        shrink=shrink,
        root_shrink=root_shrink,
        joint=bool(joint),
        verify=bool(verify),
        verify_halflife_days=verify_halflife_days,
        null_tol=null_tol,
    )
    logger.info(
        "RECAL calendar: key=%s (requested %s), weekend_dows=%s, weekly_score=%.2f (train year %s, %d seen "
        "nodes), test year %s, ticks_per_day=%d, slot_ticks=%d",
        key,
        requested,
        weekend,
        score,
        train_year,
        len(train_nodes),
        test_year,
        ticks_per_day,
        slot_ticks,
    )
    return dict(calendar=calendar, memory=memory)
