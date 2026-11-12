"""Scenario-agnostic helpers that turn time features into a ``CalendarIndex``.

Every scenario feeds the calendar memory the same way:

1. decode per-step time-of-day / day-of-week features of the *whole* series
   (fractions in [0, 1) as produced by LargeST/BasicTS-style preprocessing, or
   integer ticks) with :func:`decode_tod` / :func:`decode_dow`, or synthesize
   them with :func:`synthetic_calendar` when a dataset has no time channels;
2. reduce the pre-test data to one node-averaged series with
   :func:`per_step_series` (missing entries and network-wide outages excluded);
3. decide the calendar key on pre-test data only: :func:`select_calendar_key`
   infers the two weekend-like days and keeps the day-type split only when the
   series shows a weekly structure that replicates across the two halves of the
   span (cross-fitted score, free of the weekend-selection bias);
4. align the evaluation stream with :func:`build_calendar_index` (or do 3+4 in
   one call with :func:`calendar_from_series`, which refuses spans that reach
   into the evaluation stream) and check it with :func:`check_stream_alignment`;
5. convert the self-verification half-life (days) into a per-diagonal discount
   with :func:`verify_decay_from_halflife`.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from .calendar_memory import CalendarIndex, CalendarMemoryConfig, resolve_memory_key

logger = logging.getLogger(__name__)

# Preset frozen on the PatchSTG/SD validation stream (0.998 per diagonal at 96 ticks/day).
DEFAULT_VERIFY_HALFLIFE_DAYS = 3.6
# Minimum share of time-of-day slots observed on all seven weekdays for key selection.
MIN_COVERED_SLOTS = 0.5


def decode_tod(
    tod_values: np.ndarray,
    ticks_per_day: Optional[int] = None,
    check_contiguous: bool = True,
) -> Tuple[np.ndarray, int]:
    """Integer time-of-day tick ``[T]`` in ``[0, ticks_per_day)`` and ``ticks_per_day``.

    Accepts fractions of a day in ``[0, 1)`` or integer ticks ``0..ticks_per_day-1``.
    When ``ticks_per_day`` is not given it is inferred from the number of distinct
    values.  ``check_contiguous`` rejects encodings that do not advance by one tick
    per step (e.g. minute-of-day or float hours), which would silently alias slots.
    """
    tod = np.asarray(tod_values, dtype=np.float64).reshape(-1)
    is_fraction = float(np.nanmax(tod)) < 1.0 + 1e-6
    if ticks_per_day is None:
        ticks_per_day = int(len(np.unique(np.round(tod, 6))))
    ticks = np.rint(tod * ticks_per_day) if is_fraction else np.rint(tod)
    ticks = np.mod(ticks.astype(np.int64), ticks_per_day)
    if check_contiguous and len(ticks) > 1:
        advancing = float(np.mean(np.mod(np.diff(ticks), ticks_per_day) == 1))
        if advancing < 0.99:
            raise ValueError(
                f"time-of-day feature does not advance one tick per step ({advancing:.1%}); "
                f"check its encoding (ticks_per_day={ticks_per_day})"
            )
    return ticks, int(ticks_per_day)


def decode_dow(dow_values: np.ndarray) -> np.ndarray:
    """Integer day-of-week ``[T]`` in ``[0, 7)`` from fractions ``k/7`` or integers."""
    dow = np.asarray(dow_values, dtype=np.float64).reshape(-1)
    if 0 < float(np.nanmax(dow)) < 1.0 - 1e-6:
        dow = dow * 7.0
    return np.mod(np.rint(dow).astype(np.int64), 7)


def synthetic_calendar(
    num_steps: int, ticks_per_day: int, first_tick_of_day: int = 0, first_dow: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Calendar of a contiguous series without time channels (dow relative if unknown)."""
    absolute = np.arange(num_steps, dtype=np.int64) + int(first_tick_of_day)
    tod = absolute % ticks_per_day
    dow = (absolute // ticks_per_day + int(first_dow)) % 7
    return tod, dow


def per_step_series(values: np.ndarray, null_tol: float = 0.0) -> np.ndarray:
    """Node-averaged series ``[T]``; entries with ``|v| <= null_tol`` or non-finite are missing.

    Steps without any observed node (network-wide outages) become NaN.  A 1-D input is
    treated as an already reduced series (NaN = missing).  Pass values in original
    units: after z-scoring, missing zeros are no longer zero.
    """
    v = np.asarray(values)
    if v.ndim == 1:
        v = v.astype(np.float64)
        return np.where(np.isfinite(v), v, np.nan)
    v = v.reshape(v.shape[0], -1)
    out = np.empty(v.shape[0], dtype=np.float64)
    for lo in range(0, v.shape[0], 2048):  # chunked: bounded temporary memory
        block = v[lo : lo + 2048].astype(np.float64)
        valid = np.isfinite(block) & (np.abs(block) > null_tol)
        counts = valid.sum(1)
        sums = np.where(valid, block, 0.0).sum(1)
        out[lo : lo + 2048] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return out


def estimate_ticks_per_day(
    values: np.ndarray,
    candidates: Sequence[int] = (24, 48, 96, 144, 288),
    null_tol: float = 0.0,
) -> int:
    """Sanity check of the sampling rate: the candidate period with the most prominent
    autocorrelation peak, ``acf(L) - acf(L // 2)`` (a period-L cycle peaks at L and dips at
    L/2; smooth, persistent series such as wind power also correlate at short lags, so the
    raw autocorrelation alone is misleading).  Outages are mean-filled."""
    x = per_step_series(values, null_tol)
    fill = np.nanmean(x) if np.isfinite(x).any() else 0.0
    x = np.where(np.isfinite(x), x, fill)
    x = (x - x.mean()) / max(x.std(), 1e-12)

    def acf(lag: int) -> float:
        return float(np.mean(x[:-lag] * x[lag:])) if lag > 0 else 1.0

    scores = {
        int(L): acf(int(L)) - acf(int(L) // 2)
        for L in candidates
        if 1 < L < len(x) // 2
    }
    if not scores:
        raise ValueError("series too short for the candidate periods")
    return max(scores, key=scores.get)


def _dow_profiles(
    series: np.ndarray, tod: np.ndarray, dow: np.ndarray, ticks_per_day: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean daily profile per weekday ``[7, ticks_per_day]`` over observed steps and the
    mask of time-of-day slots observed on all seven weekdays."""
    ok = np.isfinite(series)
    prof = np.zeros((7, ticks_per_day))
    cnt = np.zeros((7, ticks_per_day))
    np.add.at(prof, (dow[ok], tod[ok]), series[ok])
    np.add.at(cnt, (dow[ok], tod[ok]), 1.0)
    covered = (cnt > 0).all(0)
    if covered.mean() < MIN_COVERED_SLOTS:
        raise ValueError(
            "pre-test data must cover all seven days of the week (observed on "
            f"{covered.mean():.0%} of the time-of-day slots)"
        )
    return prof / np.maximum(cnt, 1.0), covered


def _infer_weekend(prof: np.ndarray, covered: np.ndarray) -> Tuple[int, int]:
    p = prof[:, covered]
    dist = np.sqrt(((p - np.median(p, axis=0)) ** 2).mean(1))
    pair = max(range(7), key=lambda d: dist[d] + dist[(d + 1) % 7])
    return int(pair), int((pair + 1) % 7)


def _weekly_score(
    prof: np.ndarray, covered: np.ndarray, weekend: Sequence[int]
) -> float:
    p = prof[:, covered]
    we = sorted(set(int(x) for x in weekend))
    wk = [d for d in range(7) if d not in we]
    weekday_mean = p[wk].mean(0)
    between = np.sqrt(((p[we] - weekday_mean) ** 2).mean())
    within = np.sqrt(((p[wk] - weekday_mean) ** 2).mean())
    return float(between / max(within, 1e-12))


def _split_score(
    series: np.ndarray,
    tod: np.ndarray,
    dow: np.ndarray,
    ticks_per_day: int,
    parts: Tuple[np.ndarray, np.ndarray],
) -> float:
    try:
        first = _dow_profiles(
            series[parts[0]], tod[parts[0]], dow[parts[0]], ticks_per_day
        )
        second = _dow_profiles(
            series[parts[1]], tod[parts[1]], dow[parts[1]], ticks_per_day
        )
    except ValueError:
        return float("nan")
    return min(
        _weekly_score(*second, _infer_weekend(*first)),
        _weekly_score(*first, _infer_weekend(*second)),
    )


def _cross_fitted_score(
    series: np.ndarray, tod: np.ndarray, dow: np.ndarray, ticks_per_day: int
) -> float:
    """Weekly score of the inference procedure, free of the weekend-selection bias.

    A weekend pair is inferred on one part of the span and scored on the other (both
    directions, the smaller score is kept), so the score measures whether an inferred
    day-type effect replicates.  Persistent weather-like episodes make *some* consecutive
    pair look deviant on any span, but not the same pair on both parts.  The parts are the
    two halves (whole days); when outages leave a half without some weekday, odd and even
    weeks are used instead.  NaN when neither split covers every weekday in both parts.
    """
    steps = np.arange(len(series))
    half = (len(series) // ticks_per_day // 2) * ticks_per_day
    score = _split_score(series, tod, dow, ticks_per_day, (steps < half, steps >= half))
    if np.isnan(score):
        week = steps // ticks_per_day // 7
        score = _split_score(
            series, tod, dow, ticks_per_day, (week % 2 == 0, week % 2 == 1)
        )
    return score


def _check_weekend(weekend: Sequence[int]) -> Tuple[int, ...]:
    days = tuple(int(d) for d in weekend)
    if (
        not days
        or len(set(days)) != len(days)
        or any(d < 0 or d > 6 for d in days)
        or len(days) > 6
    ):
        raise ValueError(
            f"weekend days must be distinct integers in [0, 6], got {weekend!r}"
        )
    return days


def infer_weekend_dows(
    values: np.ndarray,
    tod: np.ndarray,
    dow: np.ndarray,
    ticks_per_day: int,
    null_tol: float = 0.0,
) -> Tuple[int, int]:
    """The two consecutive weekday indices whose daily profile deviates most.

    Works for flow and speed data alike (it uses profile distance, not level).
    ``values``: pre-test series ``[T, N]`` (or a reduced ``[T]``) aligned with ``tod``/``dow``.
    """
    prof, covered = _dow_profiles(
        per_step_series(values, null_tol), tod, dow, ticks_per_day
    )
    return _infer_weekend(prof, covered)


def weekly_structure_score(
    values: np.ndarray,
    tod: np.ndarray,
    dow: np.ndarray,
    ticks_per_day: int,
    weekend: Sequence[int],
    null_tol: float = 0.0,
) -> float:
    """Weekend-vs-weekday profile distance relative to the spread among weekdays."""
    prof, covered = _dow_profiles(
        per_step_series(values, null_tol), tod, dow, ticks_per_day
    )
    return _weekly_score(prof, covered, _check_weekend(weekend))


def select_calendar_key(
    values: np.ndarray,
    tod: np.ndarray,
    dow: np.ndarray,
    ticks_per_day: int,
    threshold: float = 2.0,
    null_tol: float = 0.0,
    weekend_override: Optional[Sequence[int]] = None,
) -> Tuple[str, Tuple[int, ...], float]:
    """``('tod_daytype', weekend, score)`` if the pre-test series has a weekly day-type
    structure (score >= threshold), else ``('tod', weekend, score)``.

    The weekend pair is inferred on the whole span.  The decision uses the cross-fitted
    score of the inference procedure (:func:`_cross_fitted_score`; NaN, hence ``'tod'``,
    when the span cannot be split into two parts that cover every weekday).  A given
    ``weekend_override`` carries no selection bias and is scored on the whole span.
    """
    tod = np.asarray(tod, dtype=np.int64)
    dow = np.asarray(dow, dtype=np.int64)
    series = per_step_series(values, null_tol)
    prof, covered = _dow_profiles(series, tod, dow, ticks_per_day)
    if weekend_override:
        weekend = _check_weekend(weekend_override)
        score = _weekly_score(prof, covered, weekend)
    else:
        weekend = _infer_weekend(prof, covered)
        score = _cross_fitted_score(series, tod, dow, ticks_per_day)
        if np.isnan(score):
            logger.warning(
                "weekly structure cannot be cross-fitted on this span (a part misses a "
                "weekday); using the time-of-day key"
            )
    return ("tod_daytype" if score >= threshold else "tod"), weekend, score


def check_stream_alignment(
    series: np.ndarray,
    first_input_step: int,
    seq_len: int,
    y_windows: np.ndarray,
    probes: Optional[Sequence[int]] = None,
    atol: float = 1e-3,
) -> None:
    """Assert that ``y_windows[p, h] == series[first_input_step + p + seq_len + h]``.

    ``series``: whole series ``[T, N]`` in the same units as ``y_windows`` ``[S, H, N]``
    (NaNs compare equal).  Call it once per scenario before trusting a calendar built
    with ``first_input_step``; reversed (legacy) streams fail at the first probe.
    """
    y = np.asarray(y_windows)
    s = np.asarray(series)
    count = y.shape[0]
    probes = probes if probes is not None else (0, 1, 2, count // 2, -2, -1)
    for p in probes:
        p = p % count
        steps = first_input_step + p + seq_len + np.arange(y.shape[1])
        if steps[-1] >= len(s) or not np.allclose(
            s[steps], y[p], atol=atol, equal_nan=True
        ):
            raise AssertionError(
                f"stream misaligned at sample {p}: labels do not match series steps "
                f"{steps[0]}..{steps[-1]}"
            )


def verify_decay_from_halflife(halflife_days: float, ticks_per_day: int) -> float:
    """Per-diagonal discount whose half-life is ``halflife_days`` days."""
    return float(0.5 ** (1.0 / (halflife_days * ticks_per_day)))


def build_calendar_index(
    tod: np.ndarray,
    dow: np.ndarray,
    ticks_per_day: int,
    key: str,
    first_input_step: int,
    num_samples: int,
    seq_len: int,
    horizon: int,
    weekend_dows: Sequence[int],
    slot_ticks: int = 1,
    target_offset: int = 1,
) -> CalendarIndex:
    """Calendar index of one evaluation stream.

    ``tod``/``dow`` cover the whole series (absolute steps).  Sample ``p`` of the
    stream reads inputs ``first_input_step + p + [0, seq_len)`` and targets
    ``first_input_step + p + seq_len + [0, horizon)``; the stream therefore spans
    ``num_samples + seq_len + horizon - 1`` steps starting at ``first_input_step``.
    ``slot_ticks > 1`` merges consecutive ticks into one time-of-day slot.
    """
    if ticks_per_day % slot_ticks:
        raise ValueError("slot_ticks must divide ticks_per_day")
    num_steps = int(num_samples) + int(seq_len) + int(horizon) - 1
    lo, hi = int(first_input_step), int(first_input_step) + num_steps
    if lo < 0 or hi > len(tod) or hi > len(dow):
        raise IndexError(
            f"stream steps [{lo}, {hi}) exceed the series length {len(tod)}"
        )
    slots = np.asarray(tod[lo:hi]) // slot_ticks
    return CalendarIndex(
        torch.as_tensor(slots, dtype=torch.long),
        torch.as_tensor(np.asarray(dow[lo:hi]), dtype=torch.long),
        ticks_per_day // slot_ticks,
        key,
        weekend_dows=_check_weekend(weekend_dows),
        input_len=seq_len,
        target_offset=target_offset,
    )


def calendar_from_series(
    values: np.ndarray,
    tod: np.ndarray,
    dow: np.ndarray,
    ticks_per_day: int,
    requested_key: str,
    select_span: slice,
    eval_first_input_step: int,
    eval_num_samples: int,
    seq_len: int,
    horizon: int,
    slot_ticks: int = 1,
    target_offset: int = 1,
    weekend_override: Optional[Sequence[int]] = None,
    threshold: float = 2.0,
    null_tol: float = 0.0,
):
    """One-call calendar construction shared by all scenarios.

    ``values``/``tod``/``dow`` cover the whole series (``values`` may be a reduced
    ``[T]`` series from :func:`per_step_series`); ``select_span`` must lie strictly
    before the evaluation stream.  Returns ``(CalendarIndex, key, weekend_dows, weekly_score)``.
    """
    lo, hi, step = select_span.indices(len(tod))
    if step != 1 or lo >= hi or hi > eval_first_input_step:
        raise ValueError(
            f"key selection span [{lo}, {hi}) must be non-empty and end before the "
            f"evaluation stream starts at step {eval_first_input_step}"
        )
    auto_key, weekend, score = select_calendar_key(
        values[lo:hi],
        tod[lo:hi],
        dow[lo:hi],
        ticks_per_day,
        threshold=threshold,
        null_tol=null_tol,
        weekend_override=weekend_override,
    )
    key = resolve_memory_key(requested_key, auto_key)
    calendar = build_calendar_index(
        tod,
        dow,
        ticks_per_day,
        key,
        eval_first_input_step,
        eval_num_samples,
        seq_len,
        horizon,
        weekend,
        slot_ticks=slot_ticks,
        target_offset=target_offset,
    )
    return calendar, key, tuple(weekend), score


def memory_config_from_options(
    key: str,
    ticks_per_day: int,
    decay: float = 0.9,
    shrink: float = 4.0,
    joint: bool = True,
    verify: bool = True,
    verify_halflife_days: float = DEFAULT_VERIFY_HALFLIFE_DAYS,
    null_tol: float = 0.0,
    root_shrink: Optional[float] = None,
) -> CalendarMemoryConfig:
    """Scenario preset -> config; the verification discount is set per tick from days."""
    return CalendarMemoryConfig(
        key=key,
        decay=decay,
        shrink=shrink,
        root_shrink=root_shrink,
        joint=joint,
        verify=verify,
        verify_decay=verify_decay_from_halflife(verify_halflife_days, ticks_per_day),
        null_tol=null_tol,
    )
