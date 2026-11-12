"""Calendar source of continual-learning (EAC) test streams for RECAL's calendar memory.

Valid only under the strict_causal protocol (chronological sample order).  The
upstream legacy FastData windows are generated in reverse time order; the
alignment check below rejects them.

Continual datasets carry no time channels, so the calendar is synthesized from
the step index: only the *consistency* of the time-of-day phase and of the
weekday cycle matters to the memory (an unknown absolute offset is harmless),
and the weekend pair is inferred from pre-test data.
"""

from recal import (
    calendar_from_series,
    check_stream_alignment,
    estimate_ticks_per_day,
    memory_config_from_options,
    per_step_series,
    synthetic_calendar,
)

MIN_SELECT_DAYS = 28


def continual_calendar_memory(
    args,
    year_series,
    train_span,
    test_first_input_step,
    test_y,
    x_len,
    y_len,
    ticks_per_day,
    logger,
):
    """Return ``(calendar, memory_config)`` for one year's test stream, or ``(None, None)``.

    Args:
        year_series: that year's series ``[T, N]`` in the label units, chronological.
        train_span: ``slice`` of ``year_series`` steps used for training (pre-test).
        test_first_input_step: index in ``year_series`` of test sample 0's first input step.
        test_y: test label windows ``[S, y_len, N]`` in stream order.
        ticks_per_day: sampling rate (288 for 5-min data).
    """
    requested = getattr(args, "recal_memory", "off")
    if requested == "off":
        return (None, None)
    null_tol = getattr(args, "recal_memory_null_tol", 0.0)
    check_stream_alignment(year_series, test_first_input_step, x_len, test_y)
    pre_test = year_series[:test_first_input_step]
    estimated = estimate_ticks_per_day(pre_test, null_tol=null_tol)
    if estimated != ticks_per_day:
        logger.warning(
            "ticks_per_day=%d but the autocorrelation suggests %d; check the raw data",
            ticks_per_day,
            estimated,
        )
    tod, dow = synthetic_calendar(len(year_series), ticks_per_day)
    lo, hi, _ = train_span.indices(test_first_input_step)
    span = (
        train_span
        if hi - lo >= MIN_SELECT_DAYS * ticks_per_day
        else slice(0, test_first_input_step)
    )
    weekend = getattr(args, "recal_memory_weekend", "")
    calendar, key, weekend, score = calendar_from_series(
        per_step_series(pre_test, null_tol),
        tod,
        dow,
        ticks_per_day,
        requested,
        span,
        test_first_input_step,
        len(test_y),
        x_len,
        y_len,
        slot_ticks=getattr(args, "recal_memory_slot_ticks", 1),
        target_offset=getattr(args, "recal_target_offset", 1),
        weekend_override=(
            tuple((int(d) for d in weekend.split(","))) if weekend else None
        ),
    )
    memory = memory_config_from_options(
        key,
        ticks_per_day,
        decay=getattr(args, "recal_memory_decay", 0.9),
        shrink=getattr(args, "recal_memory_shrink", 4.0),
        root_shrink=getattr(args, "recal_memory_root_shrink", None),
        joint=bool(getattr(args, "recal_memory_joint", 1)),
        verify=bool(getattr(args, "recal_memory_verify", 1)),
        verify_halflife_days=getattr(args, "recal_memory_verify_halflife_days", 3.6),
        null_tol=null_tol,
    )
    logger.info(
        "RECAL calendar (year %s): key=%s (requested %s), relative weekend=%s, weekly_score=%.2f, ticks_per_day=%d",
        getattr(args, "year", "?"),
        key,
        requested,
        weekend,
        score,
        ticks_per_day,
    )
    return (calendar, memory)
