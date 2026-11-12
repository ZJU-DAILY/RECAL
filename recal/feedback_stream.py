"""Label-arrival trace sources for asynchronous RECAL replay.

``FeedbackTraceSource`` owns the (event_time, node) -> (value, arrival_time)
table and hands out label events strictly by processing tick.  Engines call
``pop_through(t)`` once per tick and forward the returned events to
``ForecastDiagonalAssimilator.observe``; the maintenance layer never sees the
oracle itself, so it cannot peek at a label before its arrival tick.

``arrival[e, n]`` stores the arrival tick of the label for target time ``e`` and
node ``n``.  ``NEVER`` (-1) marks labels that never arrive.  A stored value of
0.0 is a valid observation; arrival and loss masking are independent.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

NEVER = -1


class FeedbackTraceSource:
    """Replayable label-event oracle sorted by (arrival, event_time, node)."""

    def __init__(
        self,
        values: torch.Tensor,
        arrival: torch.Tensor,
        present: Optional[torch.Tensor] = None,
    ) -> None:
        values = torch.as_tensor(values, dtype=torch.float32)
        arrival = torch.as_tensor(arrival, dtype=torch.long)
        if values.shape != arrival.shape or values.dim() != 2:
            raise ValueError(
                "values and arrival must both be [num_event_times, num_nodes]"
            )
        if present is None:
            present = torch.ones_like(arrival, dtype=torch.bool)
        present = torch.as_tensor(present, dtype=torch.bool)
        if present.shape != arrival.shape:
            raise ValueError("present mask must match the [E, N] trace shape")

        valid = present & (arrival != NEVER)
        event_index = torch.arange(arrival.shape[0]).unsqueeze(-1).expand_as(arrival)
        if bool((arrival[valid] < event_index[valid]).any()):
            raise ValueError(
                "causality violation: some arrival precedes its event time"
            )

        event_ids, node_ids = valid.nonzero(as_tuple=True)
        arr = arrival[event_ids, node_ids].numpy()
        tau = event_ids.numpy()
        node = node_ids.numpy()
        val = values[event_ids, node_ids].numpy()
        order = np.lexsort((node, tau, arr))  # primary: arrival, then tau, then node
        self._arr = arr[order]
        self._tau = tau[order]
        self._node = node[order]
        self._val = val[order]
        self._cursor = 0
        self.num_event_times, self.num_nodes = arrival.shape

    # ------------------------------------------------------------------- popping
    def pop_through(
        self, processing_time: int
    ) -> List[Tuple[int, np.ndarray, torch.Tensor, int]]:
        """Return un-popped events with ``arrival <= processing_time``.

        Events are grouped per (arrival, target_time) call -- ready to forward to
        ``observe(target_time, node_ids, values, arrival_time=arrival)`` -- and
        emitted in ascending (arrival, target_time) order.  Repeated calls never
        re-emit an event (pop semantics).
        """
        end = int(np.searchsorted(self._arr, int(processing_time), side="right"))
        events: List[Tuple[int, np.ndarray, torch.Tensor, int]] = []
        i = self._cursor
        while i < end:
            arr_i, tau_i = self._arr[i], self._tau[i]
            j = i
            while j < end and self._arr[j] == arr_i and self._tau[j] == tau_i:
                j += 1
            events.append(
                (
                    int(tau_i),
                    self._node[i:j].copy(),
                    torch.from_numpy(self._val[i:j].copy()),
                    int(arr_i),
                )
            )
            i = j
        self._cursor = max(self._cursor, end)
        return events

    @property
    def remaining(self) -> int:
        return len(self._arr) - self._cursor

    # ---------------------------------------------------------------- builders
    @classmethod
    def on_time(
        cls,
        values: torch.Tensor,
        present: Optional[torch.Tensor] = None,
    ) -> "FeedbackTraceSource":
        """F0 trace: every present label arrives exactly at its event time."""
        values = torch.as_tensor(values, dtype=torch.float32)
        num_events, num_nodes = values.shape
        arrival = (
            torch.arange(num_events, dtype=torch.long)
            .unsqueeze(-1)
            .expand(num_events, num_nodes)
        )
        return cls(values, arrival.clone(), present=present)

    @classmethod
    def from_arrays(
        cls,
        num_event_times: int,
        num_nodes: int,
        target_time: np.ndarray,
        node_id: np.ndarray,
        arrival_time: np.ndarray,
        values_table: torch.Tensor,
    ) -> "FeedbackTraceSource":
        """Build from a flat trace bundle (target_time, node_id, arrival_time).

        ``arrival_time == NEVER`` marks permanently missing labels; values are
        injected from the canonical ``values_table`` [E, N] at arrival time.
        """
        arrival = torch.full(
            (int(num_event_times), int(num_nodes)), NEVER, dtype=torch.long
        )
        tt = torch.as_tensor(np.asarray(target_time), dtype=torch.long)
        nid = torch.as_tensor(np.asarray(node_id), dtype=torch.long)
        arr = torch.as_tensor(np.asarray(arrival_time), dtype=torch.long)
        arrival[tt, nid] = arr
        return cls(torch.as_tensor(values_table, dtype=torch.float32), arrival)


def lattice_truth_table(
    truth: torch.Tensor,
    target_offset: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a captured lattice truth tensor [T, H, N, 1] to an event-time table.

    Returns ``(values [E, N], present [E, N])`` where row ``tau`` holds the label
    of event time ``tau = origin + horizon + target_offset``.  Rows before the
    first reachable event time are marked absent.
    """
    truth_t = torch.as_tensor(truth)[..., 0].to(torch.float32)  # [T, H, N]
    num_origins, pred_len, num_nodes = truth_t.shape
    num_events = num_origins + pred_len + target_offset - 1
    table = torch.zeros(num_events, num_nodes, dtype=torch.float32)
    present = torch.zeros(num_events, num_nodes, dtype=torch.bool)
    for horizon in range(pred_len):
        lo = target_offset + horizon
        table[lo : lo + num_origins] = truth_t[:, horizon, :]
        present[lo : lo + num_origins] = True
    return table, present


class StreamingTraceFeeder:
    """Engine-side oracle: labels stream in with the loader, events pop by arrival.

    The four scenario engines receive labels batch-by-batch rather than as a
    pre-built table.  This feeder accumulates each newly seen event-time row
    (first writer wins; lattices are consistent across origins), materializes
    its label events according to an arrival schedule, and releases them
    strictly by arrival tick.  The assimilator never touches this object.

    ``arrival`` is an optional [E, N] long tensor (``NEVER`` = label never
    arrives); ``None`` means on-time delivery (arrival == event time, F0).
    """

    def __init__(
        self,
        num_nodes: int,
        arrival: Optional[torch.Tensor] = None,
        target_offset: int = 1,
    ) -> None:
        self.num_nodes = int(num_nodes)
        self.target_offset = int(target_offset)
        self.arrival = (
            None if arrival is None else torch.as_tensor(arrival, dtype=torch.long)
        )
        self._seen_rows: set = set()
        self._heap: List[Tuple[int, int, np.ndarray, np.ndarray]] = (
            []
        )  # (arr, tau, nodes, values)

    def push_labels(self, origin_start: int, label: torch.Tensor) -> None:
        """Feed one batch of loader labels ``[B, H, N, 1]`` (original scale)."""
        label = torch.as_tensor(label).detach()
        batch_size, pred_len = label.shape[0], label.shape[1]
        for batch_idx in range(batch_size):
            origin = int(origin_start) + batch_idx
            for horizon in range(pred_len):
                tau = origin + horizon + self.target_offset
                if tau in self._seen_rows:
                    continue
                self._seen_rows.add(tau)
                row = label[batch_idx, horizon, :, 0].to(torch.float32).cpu()
                self._materialize(tau, row)

    def _materialize(self, tau: int, row: torch.Tensor) -> None:
        if self.arrival is None:
            nodes = np.arange(self.num_nodes)
            heapq.heappush(self._heap, (tau, tau, nodes, row.numpy().copy()))
            return
        arr_row = self.arrival[tau]
        for arr in torch.unique(arr_row).tolist():
            if arr == NEVER:
                continue
            if arr < tau:
                raise ValueError(
                    f"causality violation in arrival schedule at tau={tau}: {arr} < {tau}"
                )
            nodes = (arr_row == arr).nonzero(as_tuple=True)[0].numpy()
            heapq.heappush(
                self._heap, (int(arr), tau, nodes, row[nodes].numpy().copy())
            )

    def pop_through(
        self, processing_time: int
    ) -> List[Tuple[int, np.ndarray, torch.Tensor, int]]:
        events: List[Tuple[int, np.ndarray, torch.Tensor, int]] = []
        while self._heap and self._heap[0][0] <= int(processing_time):
            arr, tau, nodes, values = heapq.heappop(self._heap)
            events.append((tau, nodes, torch.from_numpy(values), arr))
        return events
