"""Backward-compatible facade for the accepted RECAL update utilities.

The implementation lives in cohesive modules (calibrator / controller /
maintenance / event_maintenance / feedback_stream). This module re-exports the
public surface so existing imports ``from recal.online_updates import X`` and
``from recal import X`` keep working unchanged.
"""

from __future__ import annotations

from .calibrator import (
    ReCalibrator,
    TensorLoss,
    count_trainable_params,
    masked_mae_loss,
    select_horizon_values,
    select_raw_horizon_values,
)
from .controller import EventFeedbackStats, FeedbackCell, OnlineUpdateStats
from .feedback_stream import (
    NEVER,
    FeedbackTraceSource,
    StreamingTraceFeeder,
    lattice_truth_table,
)
from .maintenance import (
    EVENT_TIME_FEEDBACK,
    FEEDBACK_MODES,
    LEGACY_FEEDBACK,
    ForecastDiagonalAssimilator,
    MaintenanceOperator,
)

__all__ = [
    "ReCalibrator",
    "EVENT_TIME_FEEDBACK",
    "FEEDBACK_MODES",
    "TensorLoss",
    "EventFeedbackStats",
    "FeedbackCell",
    "FeedbackTraceSource",
    "LEGACY_FEEDBACK",
    "NEVER",
    "OnlineUpdateStats",
    "ForecastDiagonalAssimilator",
    "MaintenanceOperator",
    "StreamingTraceFeeder",
    "lattice_truth_table",
    "masked_mae_loss",
    "select_horizon_values",
    "select_raw_horizon_values",
    "count_trainable_params",
]
