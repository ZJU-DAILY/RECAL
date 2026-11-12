"""Bounded asynchronous feedback for RECAL.

A forecast issued at ``origin`` for horizon ``h`` predicts target time
``tau = origin + h + target_offset``.  Its node labels may arrive later.  The
engine follows one rule: use each arrived label once if it arrives no later than
``tau + allowed_lateness``; discard any unresolved state at that deadline.

``issue`` stores forecasts, ``observe`` accepts arrived labels, ``maintain``
performs ready updates in target-time order, and ``finalize`` only clears state.
The engine never owns future labels.  A value of 0.0 is still a valid arrival;
the project-wide masked loss decides whether that value contributes to loss.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Set, Tuple, Union

import torch

from .controller import EventFeedbackStats, FeedbackCell

NodeIds = Union[Sequence[int], torch.Tensor]
Values = Union[Sequence[float], torch.Tensor]


class _PendingGroup:
    """All issued cells for one target time, plus its per-node label state."""

    __slots__ = ("cells", "values", "consumed", "num_nodes")

    def __init__(self, num_nodes: int) -> None:
        self.num_nodes = int(num_nodes)
        self.cells: List[Tuple[int, int, torch.Tensor]] = (
            []
        )  # (origin, horizon, raw_pred)
        self.values = torch.zeros(self.num_nodes, dtype=torch.float32)
        self.consumed = torch.zeros(self.num_nodes, dtype=torch.bool)


class EventTimeEngine:
    """Match arrived node labels to pending forecasts within a finite wait."""

    def __init__(self, assimilator, allowed_lateness: int = 0) -> None:
        self.assim = assimilator
        self.allowed_lateness = max(0, int(allowed_lateness))
        self.stats = EventFeedbackStats()
        self.groups: Dict[int, _PendingGroup] = {}
        # tau -> node -> (value, arrival_time); first delivery wins.
        self._inbox: Dict[int, Dict[int, Tuple[float, int]]] = {}
        self._max_issued_tau = -1
        self._released_through = -1  # all tau <= this are released
        self._early_released: Set[int] = set()  # released above the watermark

    # ------------------------------------------------------------------ issue
    def issue(self, origin_start: int, raw_pred: torch.Tensor) -> None:
        raw_pred = raw_pred.detach()
        batch_size, _, num_nodes, pred_len = raw_pred.shape
        for batch_idx in range(batch_size):
            origin = int(origin_start) + batch_idx
            raw_one = raw_pred[batch_idx : batch_idx + 1]
            for horizon in range(pred_len):
                target_time = origin + horizon + self.assim.target_offset
                group = self.groups.get(target_time)
                if group is None:
                    if self._is_released(target_time):
                        raise RuntimeError(
                            f"issue(): target_time {target_time} was already released; "
                            "predictions must be issued before their labels can mature"
                        )
                    group = _PendingGroup(num_nodes)
                    self.groups[target_time] = group
                elif bool(group.consumed.any()):
                    raise RuntimeError(
                        f"issue(): target_time {target_time} is partially consumed; "
                        "late issue into a consumed diagonal violates exactly-once"
                    )
                group.cells.append((origin, horizon, raw_one))
                if target_time > self._max_issued_tau:
                    self._max_issued_tau = target_time
        self.stats.note_working_set(*self.working_set())

    # ---------------------------------------------------------------- observe
    def observe(
        self, target_time: int, node_ids: NodeIds, values: Values, *, arrival_time: int
    ) -> None:
        tau = int(target_time)
        arrival = int(arrival_time)
        if arrival < tau:
            raise ValueError(
                f"causality violation: arrival_time {arrival} < target_time {tau}"
            )
        vals = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
        ids = torch.as_tensor(node_ids, dtype=torch.long).reshape(-1)
        if ids.numel() != vals.numel():
            raise ValueError("observe(): node_ids and values length mismatch")
        if not bool(torch.isfinite(vals).all()):
            raise ValueError("observe(): non-finite label value in feedback event")

        group = self.groups.get(tau)
        if group is None:
            count = int(ids.numel())
            if tau > self._max_issued_tau:
                self.stats.orphan += count  # no prediction ever issued for tau
            elif tau in self._early_released:
                # Fully consumed and released: bounded state forbids keeping the
                # values, so re-deliveries can no longer be value-checked.
                self.stats.duplicate += count
            else:
                self.stats.late += count  # released (deadline passed) before delivery
            return
        if arrival > tau + self.allowed_lateness:
            self.stats.late += int(ids.numel())
            return

        pending = self._inbox.setdefault(tau, {})
        for node, value in zip(ids.tolist(), vals.tolist()):
            if node < 0 or node >= group.num_nodes:
                raise ValueError(
                    f"observe(): node id {node} out of range [0, {group.num_nodes})"
                )
            if bool(group.consumed[node]):
                if math.isclose(
                    value, float(group.values[node]), rel_tol=1e-6, abs_tol=1e-6
                ):
                    self.stats.duplicate += 1
                    continue
                self.stats.conflict += 1
                raise ValueError(
                    f"observe(): conflicting re-delivery for (tau={tau}, node={node})"
                )
            if node in pending:
                if math.isclose(value, pending[node][0], rel_tol=1e-6, abs_tol=1e-6):
                    self.stats.duplicate += 1  # keep the earliest arrival
                    continue
                self.stats.conflict += 1
                raise ValueError(
                    f"observe(): conflicting pending delivery for (tau={tau}, node={node})"
                )
            pending[node] = (value, arrival)
            self.stats.arrived += 1
        if not pending:
            self._inbox.pop(tau, None)

    # --------------------------------------------------------------- maintain
    def maintain(self, processing_time: int) -> None:
        now = int(processing_time)
        due_taus = sorted(
            set(self._inbox.keys())
            | {tau for tau in self.groups if tau + self.allowed_lateness <= now}
        )
        for tau in due_taus:
            group = self.groups.get(tau)
            pending = self._inbox.get(tau)
            if group is None:
                if pending:  # defensive: inbox for a released tau cannot be consumed
                    self.stats.late += len(pending)
                    self._inbox.pop(tau, None)
                continue
            if pending:
                ready = {
                    node: pending.pop(node)
                    for node in list(pending)
                    if pending[node][1] <= now
                }
                if not pending:
                    self._inbox.pop(tau, None)
                if ready:
                    self._consume_round(tau, group, ready, now)
            if tau + self.allowed_lateness <= now:
                self._evict(tau, group)  # consume-then-evict at the deadline
            elif bool(group.consumed.all()):
                self._release(tau)  # nothing left to wait for
        if now - self.allowed_lateness > self._released_through:
            self._released_through = now - self.allowed_lateness
            self._early_released = {
                t for t in self._early_released if t > self._released_through
            }
        self.stats.note_working_set(*self.working_set())

    def _consume_round(
        self,
        tau: int,
        group: _PendingGroup,
        ready: Dict[int, Tuple[float, int]],
        now: int,
    ) -> None:
        round_mask = torch.zeros(group.num_nodes, dtype=torch.bool)
        for node, (value, arrival) in ready.items():
            if arrival > now:  # unreachable by construction; hard causality guard
                self.stats.future_use += 1
                raise RuntimeError(
                    "future-use: consuming an event that has not arrived"
                )
            group.values[node] = value
            group.consumed[node] = True
            round_mask[node] = True
            self.stats.record_age(now - tau)
        self.stats.consumed += len(ready)
        target = torch.where(
            round_mask, group.values, torch.zeros_like(group.values)
        ).unsqueeze(
            -1
        )  # [N, 1]
        cells = [
            FeedbackCell(
                origin=origin,
                target_time=tau,
                horizon=horizon,
                raw_pred=raw_one,
                target=target,
                node_mask=round_mask,
            )
            for origin, horizon, raw_one in group.cells
        ]
        self.stats.rounds += 1
        self.assim._ingest_matured_diagonal(cells)

    # ----------------------------------------------------------- release paths
    def _evict(self, tau: int, group: _PendingGroup) -> None:
        unconsumed = int((~group.consumed).sum().item())
        self.stats.expired_node_slots += len(group.cells) * unconsumed
        self._release(tau)

    def _release(self, tau: int) -> None:
        self.groups.pop(tau, None)
        leftover = self._inbox.pop(tau, None)
        if leftover:  # delivered in-window but never became consumable
            self.stats.late += len(leftover)
        if tau > self._released_through:
            self._early_released.add(tau)

    def finalize(self, processing_time: int) -> None:
        """Evict all remaining state; no update may follow the last prediction."""
        del processing_time  # eviction at finalize is unconditional
        for tau in sorted(self.groups):
            self._evict(tau, self.groups[tau])
        self._inbox.clear()
        # Discard any half-merged lazy (A2) buffer without stepping.
        dropped = len(self.assim._pending_cells)
        if dropped:
            self.stats.discarded_merge_cells += dropped
        self.assim._pending_cells.clear()
        self.assim._pending_diagonals = 0
        if self._max_issued_tau > self._released_through:
            self._released_through = self._max_issued_tau
            self._early_released.clear()

    # -------------------------------------------------------------- telemetry
    def working_set(self) -> Tuple[int, int, int]:
        """Return (pending groups, pending (cell, node) slots, actual bytes)."""
        node_cells = 0
        nbytes = 0
        seen_storages: Set[int] = set()
        for group in self.groups.values():
            unconsumed = int((~group.consumed).sum().item())
            node_cells += len(group.cells) * unconsumed
            nbytes += group.values.numel() * group.values.element_size()
            nbytes += group.consumed.numel() * group.consumed.element_size()
            for _, _, raw_one in group.cells:
                key = raw_one.data_ptr()
                if key not in seen_storages:
                    seen_storages.add(key)
                    nbytes += raw_one.numel() * raw_one.element_size()
        return len(self.groups), node_cells, nbytes

    def _is_released(self, tau: int) -> bool:
        return tau <= self._released_through or tau in self._early_released
