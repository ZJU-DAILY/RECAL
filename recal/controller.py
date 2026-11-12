"""Feedback records and runtime statistics for RECAL.

``FeedbackCell`` pairs one historical forecast with supervision at its target
time.  ``node_mask`` is absent for full-diagonal fixed replay and selects the
arrived nodes for asynchronous replay.  The two statistics classes keep model
updates and asynchronous delivery accounting separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class FeedbackCell:
    origin: int
    target_time: int
    horizon: int
    raw_pred: torch.Tensor
    target: torch.Tensor
    # Event mode only: nodes that belong to this consumption round (bool [N]).
    # ``None`` in legacy fixed-replay mode (whole-diagonal consumption).
    node_mask: Optional[torch.Tensor] = None


@dataclass
class OnlineUpdateStats:
    eligible: int = 0
    applied: int = 0
    cells_sum: int = 0
    adapt_ms_sum: float = 0.0
    # A2 lazy maintenance: how many matured diagonals were merged into one step.
    merged_diagonals_sum: int = 0
    # A3 budgeted maintenance: nodes touched per step, and the dropped-tail
    # energy epsilon_K (an online-monitorable upper bound on the sparse-update
    # excess loss).
    nodes_updated_sum: int = 0
    eps_k_sum: float = 0.0

    def record(
        self,
        cells: int,
        adapt_ms: float,
        merged_diagonals: int = 1,
        nodes_updated: int = 0,
        eps_k: float = 0.0,
    ) -> None:
        self.eligible += 1
        self.applied += 1
        self.cells_sum += int(cells)
        self.adapt_ms_sum += float(adapt_ms)
        self.merged_diagonals_sum += int(merged_diagonals)
        self.nodes_updated_sum += int(nodes_updated)
        self.eps_k_sum += float(eps_k)

    def record_deferred(self, cells: int = 0) -> None:
        self.eligible += 1
        self.cells_sum += int(cells)

    @property
    def avg_cells(self) -> float:
        return self.cells_sum / self.eligible if self.eligible else 0.0

    @property
    def avg_adapt_ms(self) -> float:
        return self.adapt_ms_sum / self.applied if self.applied else 0.0

    @property
    def avg_merged_diagonals(self) -> float:
        return self.merged_diagonals_sum / self.applied if self.applied else 0.0

    @property
    def avg_nodes_updated(self) -> float:
        return self.nodes_updated_sum / self.applied if self.applied else 0.0

    @property
    def avg_eps_k(self) -> float:
        return self.eps_k_sum / self.applied if self.applied else 0.0


@dataclass
class EventFeedbackStats:
    """Observability surface for event-time (asynchronous) feedback ingestion.

    Every accepted (target_time, node) label event is classified exactly once:
    consumed / duplicate / conflict / late / orphan.  Every issued (cell, node)
    slot that never receives its label by the allowed-lateness deadline is
    counted in ``expired_node_slots``.  ``future_use`` must remain 0 by
    construction: consuming an event whose arrival lies in the future would be
    a causality violation.
    """

    arrived: int = 0  # first-arrival label events accepted into the pending buffer
    consumed: int = 0  # events whose value reached a gradient round
    duplicate: int = 0  # idempotent re-deliveries (same value)
    conflict: int = 0  # contradictory re-deliveries (also raises)
    late: int = 0  # events beyond the allowed-lateness deadline (never consumed)
    orphan: int = 0  # events for target times with no issued prediction
    future_use: int = 0  # causality guard; must stay 0
    expired_node_slots: int = 0  # (cell, node) slots evicted without supervision
    discarded_merge_cells: int = (
        0  # consumed cells dropped from the lazy (A2) buffer at finalize
    )
    rounds: int = 0  # per-target-time consumption rounds executed
    max_pending_groups: int = 0
    max_node_cells: int = 0
    max_bytes: int = 0
    age_hist: Dict[int, int] = field(default_factory=dict)

    def record_age(self, age: int, count: int = 1) -> None:
        self.age_hist[int(age)] = self.age_hist.get(int(age), 0) + int(count)

    def note_working_set(self, groups: int, node_cells: int, nbytes: int) -> None:
        self.max_pending_groups = max(self.max_pending_groups, int(groups))
        self.max_node_cells = max(self.max_node_cells, int(node_cells))
        self.max_bytes = max(self.max_bytes, int(nbytes))

    @property
    def age_mean(self) -> float:
        total = sum(self.age_hist.values())
        if total == 0:
            return 0.0
        return sum(age * cnt for age, cnt in self.age_hist.items()) / total

    def age_quantile(self, q: float = 0.95) -> int:
        total = sum(self.age_hist.values())
        if total == 0:
            return 0
        threshold = q * total
        cumulative = 0
        for age in sorted(self.age_hist):
            cumulative += self.age_hist[age]
            if cumulative >= threshold:
                return age
        return max(self.age_hist)
