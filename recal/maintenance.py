"""Feedback maintenance for the RECAL calibrator.

RECAL keeps the calibrator and the feedback policy separate.  Both supported
policies eventually produce the same unit of supervision: a target-time
diagonal of historical forecasts.

``legacy``
    Fixed replay used by the accepted main method.  Predictions and targets are
    queued together, but a target is consumed only after its target time matures.

``event_time``
    Optional asynchronous replay.  Predictions are registered with ``issue``;
    node labels enter later through ``observe``.  A label is used at most once
    when it arrives by ``target_time + allowed_lateness``; the remainder is then
    discarded.  Partial-node updates and bounded state follow from this rule.

``update_interval`` and ``maintenance_budget`` remain optional efficiency knobs.
Their defaults (1 and 0) give eager, dense updates.  The optimizer itself is a
plain fixed-step optimizer supplied by the caller.
"""

from __future__ import annotations

import abc
import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .calendar_memory import CalendarIndex, CalendarMemory, CalendarMemoryConfig
from .calibrator import masked_mae_loss, select_horizon_values
from .controller import FeedbackCell, OnlineUpdateStats
from .event_maintenance import EventTimeEngine

logger = logging.getLogger(__name__)

LEGACY_FEEDBACK = "legacy"
EVENT_TIME_FEEDBACK = "event_time"
FEEDBACK_MODES = (LEGACY_FEEDBACK, EVENT_TIME_FEEDBACK)


class MaintenanceOperator(abc.ABC):
    """Interface for a streaming forecast-maintenance operator.

    The base contract mirrors the accepted fixed-replay loop::

        operator.maintain(current_origin)   # consume matured diagonals
        y = predict(...)                     # read-path uses operator state
        operator.ingest(origin, raw_pred, label)

    In asynchronous mode, ``maintain`` is shared while ``issue``/``observe`` and
    ``finalize`` replace ``ingest`` and ``flush``.  Historical method names
    ``update_available`` and ``enqueue`` remain available for compatibility.
    """

    @abc.abstractmethod
    def ingest(
        self, origin_start: int, raw_pred: torch.Tensor, target: torch.Tensor
    ) -> None: ...

    @abc.abstractmethod
    def maintain(self, current_origin: int) -> None: ...

    @abc.abstractmethod
    def flush(self) -> None: ...


class ForecastDiagonalAssimilator(MaintenanceOperator):
    """Update from target-time diagonals of the sliding forecast lattice.

    Defaults (``update_interval=1``, ``maintenance_budget=0``) reproduce the
    eager dense A1 operator exactly. The calibrator is updated by the
    caller-supplied fixed-step ``optimizer``.
    """

    def __init__(
        self,
        calibrator: nn.Module,
        optimizer: torch.optim.Optimizer,
        mean: float,
        std: float,
        device: torch.device,
        target_offset: int = 1,
        update_interval: int = 1,
        maintenance_budget: int = 0,
        feedback_mode: str = LEGACY_FEEDBACK,
        allowed_lateness: int = 0,
        execution_backend: str = "reference",
        calendar: Optional[CalendarIndex] = None,
        memory: Optional[CalendarMemoryConfig] = None,
    ) -> None:
        self.calibrator = calibrator
        self.optimizer = optimizer
        self.mean = float(mean)
        self.std = float(std)
        self.device = (
            torch.device(device)
            if device is not None
            else next(calibrator.parameters()).device
        )
        self.target_offset = int(target_offset)
        from .execution import EXECUTION_BACKENDS, HardwareAwareUpdate

        if execution_backend not in EXECUTION_BACKENDS:
            raise ValueError(f"unknown execution_backend: {execution_backend!r}")
        self.execution_backend = execution_backend
        self._execution = (
            HardwareAwareUpdate(self) if execution_backend == "hardware_aware" else None
        )

        # A2 (lazy interval-k maintenance).
        self.update_interval = max(1, int(update_interval))
        self._pending_cells: List[FeedbackCell] = []
        self._pending_diagonals = 0

        # A3 (budgeted Top-K sparse maintenance). 0 disables (dense update).
        self.maintenance_budget = max(0, int(maintenance_budget))

        self.max_working_set = 0
        self.buckets: Dict[int, List[FeedbackCell]] = defaultdict(list)
        self.stats = OnlineUpdateStats()

        # Event-time asynchronous feedback (issue/observe) vs legacy fixed-replay.
        if feedback_mode not in FEEDBACK_MODES:
            raise ValueError(f"unknown feedback_mode: {feedback_mode!r}")
        self.feedback_mode = feedback_mode
        self.allowed_lateness = max(0, int(allowed_lateness))
        self._event = (
            EventTimeEngine(self, allowed_lateness=self.allowed_lateness)
            if feedback_mode == EVENT_TIME_FEEDBACK
            else None
        )

        # Optional calendar-addressed residual memory (off by default).
        self.calendar = None
        self.memory = None
        if memory is not None and memory.enabled:
            if calendar is None:
                raise ValueError("calendar memory requires a CalendarIndex")
            if feedback_mode != LEGACY_FEEDBACK or execution_backend != "reference":
                raise ValueError(
                    "calendar memory supports legacy feedback with reference execution only"
                )
            if self.update_interval > calendar.period:
                # Merged diagonals more than one day apart would collide on a slot.
                raise ValueError(
                    "calendar memory requires update_interval <= calendar period"
                )
            if self.update_interval > 1 and calendar.repeats_adjacent_slots():
                # Two merged diagonals on one slot would overwrite each other's residuals.
                raise ValueError(
                    "merged diagonals (update_interval > 1) need slot_ticks = 1"
                )
            self.calendar = calendar.to(self.device)
            self.memory = CalendarMemory(
                calendar.level_sizes,
                calibrator.pred_len,
                calibrator.num_nodes,
                memory,
                self.device,
                level_rates=calendar.level_rates,
            )

    # ------------------------------------------------------------------ ingest
    def enqueue(
        self, origin_start: int, raw_pred: torch.Tensor, target: torch.Tensor
    ) -> None:
        self._require_mode(LEGACY_FEEDBACK, "enqueue")
        raw_pred = raw_pred.detach()
        target = target.detach()
        batch_size, _, _, pred_len = raw_pred.shape
        for batch_idx in range(batch_size):
            origin = origin_start + batch_idx
            raw_one = raw_pred[batch_idx : batch_idx + 1]
            target_one = target[batch_idx]
            for horizon in range(pred_len):
                target_time = origin + horizon + self.target_offset
                cell = FeedbackCell(
                    origin=origin,
                    target_time=target_time,
                    horizon=horizon,
                    raw_pred=raw_one,
                    target=target_one[horizon],
                )
                self.buckets[target_time].append(cell)

        ws = sum(len(v) for v in self.buckets.values()) + len(self._pending_cells)
        if ws > self.max_working_set:
            self.max_working_set = ws

    # ----------------------------------------------------------------- maintain
    def update_available(self, current_origin: int) -> None:
        self._require_mode(LEGACY_FEEDBACK, "update_available")
        mature_keys = [key for key in self.buckets.keys() if key <= current_origin]
        for key in sorted(mature_keys):
            cells = self.buckets.pop(key)
            if cells:
                self._ingest_matured_diagonal(cells)

    def _ingest_matured_diagonal(self, cells: List[FeedbackCell]) -> None:
        """A2 trigger point. Eager when update_interval==1; otherwise accumulate
        matured diagonals and merge them into a single step every k diagonals."""
        if self.update_interval <= 1:
            self._update_cells(cells, merged_diagonals=1)
            return
        self._pending_cells.extend(cells)
        self._pending_diagonals += 1
        if self._pending_diagonals >= self.update_interval:
            self._flush_pending()

    def _flush_pending(self) -> None:
        if not self._pending_cells:
            self._pending_diagonals = 0
            return
        merged = self._pending_diagonals
        cells = self._pending_cells
        self._pending_cells = []
        self._pending_diagonals = 0
        self._update_cells(cells, merged_diagonals=max(1, merged))

    def flush(self) -> None:
        self._require_mode(LEGACY_FEEDBACK, "flush")
        for key in sorted(list(self.buckets.keys())):
            cells = self.buckets.pop(key)
            if cells:
                self._ingest_matured_diagonal(cells)
        # drain any cells still pending under lazy maintenance
        self._flush_pending()

    # ------------------------------------------------------------------- update
    def _select_topk_nodes(
        self,
        pred_orig: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ):
        """A3: pick Top-K node rows by importance ``||U[i,:]|| * ||residual_i||``.

        Returns ``(node_index, eps_k)`` or ``(None, 0.0)`` when budgeting is off
        or unnecessary. ``eps_k`` is the residual energy of the dropped tail.
        ``valid_mask`` (event mode) restricts the importance signal to nodes that
        belong to the current consumption round.
        """
        budget = self.maintenance_budget
        num_nodes = pred_orig.shape[1]
        if budget <= 0 or budget >= num_nodes:
            return None, 0.0
        with torch.no_grad():
            # per-node residual energy aggregated over the matured cells
            resid = torch.abs(pred_orig - target)  # [K, N, 1]
            if valid_mask is not None:
                resid = resid * valid_mask.unsqueeze(-1).to(resid.dtype)
            resid_node = resid.flatten(start_dim=2).mean(dim=-1).mean(dim=0)  # [N]
            loading = getattr(self.calibrator, "node_loading_norm", None)
            if callable(loading):
                importance = loading().to(resid_node.device) * resid_node
            else:
                importance = resid_node
            topk = torch.topk(importance, k=budget, largest=True)
            sel = topk.indices
            total_energy = float(torch.sum(resid_node**2).item())
            kept_energy = float(torch.sum(resid_node[sel] ** 2).item())
            eps_k = max(0.0, total_energy - kept_energy)
        return sel, eps_k

    def _mask_node_grads(self, node_index: torch.Tensor) -> None:
        """Zero gradients of per-node calibrator params outside the Top-K set.

        Only the per-node parameters (``horizon_node[N,r]``, ``node_scale[N,1]``,
        ``node_shift[N,1]``) are sparsified; the tiny shared ``r x H`` bases are
        left dense (cost is O(rH) regardless of N).
        """
        per_node_params = []
        for name in ("horizon_node", "node_scale", "node_shift"):
            param = getattr(self.calibrator, name, None)
            if (
                isinstance(param, nn.Parameter)
                and param.grad is not None
                and param.shape[0] == self.calibrator.num_nodes
            ):
                per_node_params.append(param)
        if not per_node_params:
            return
        mask = torch.zeros(self.calibrator.num_nodes, device=node_index.device)
        mask[node_index] = 1.0
        for param in per_node_params:
            view = mask.view(-1, *([1] * (param.grad.dim() - 1)))
            param.grad.mul_(view.to(param.grad.device, param.grad.dtype))

    def _update_cells(
        self, cells: List[FeedbackCell], merged_diagonals: int = 1
    ) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start_time = time.perf_counter()

        if self._execution is not None and self._execution.run(cells):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.stats.record(
                len(cells),
                (time.perf_counter() - start_time) * 1000.0,
                merged_diagonals=merged_diagonals,
                nodes_updated=self.calibrator.num_nodes,
            )
            return

        raw_batch = torch.cat([cell.raw_pred for cell in cells], dim=0).to(self.device)
        horizons = torch.tensor(
            [cell.horizon for cell in cells], device=self.device, dtype=torch.long
        )
        target = torch.stack([cell.target for cell in cells], dim=0).to(self.device)

        self.calibrator.train()
        calibrated = self.calibrator(raw_batch)
        selected_norm = select_horizon_values(calibrated, horizons)
        pred_orig = selected_norm.unsqueeze(-1) * self.std + self.mean
        depths = None
        fitted = pred_orig
        if self.memory is not None:
            target_times = [cell.target_time for cell in cells]
            offset, lag = self.calendar.target_offset, self.calendar.input_len
            self.calendar.check_steps(
                min(target_times) - offset + lag, max(target_times) - offset + lag
            )
            mem_slots = self.calendar.slots_of_targets(
                torch.tensor(target_times, device=self.device)
            )
            with torch.no_grad():
                depths = self.memory.read_depths(
                    mem_slots, horizons
                )  # [L, K, N], pre-write
            if self.memory.config.joint:
                # Backfitting: the calibrator explains what the (verified) recall does not.
                fitted = pred_orig + self.memory.served(depths).unsqueeze(-1)
        loss = masked_mae_loss(fitted, target)

        # A3: pick Top-K node subset (free U-driven importance) before backward.
        valid_mask = None
        if cells and all(cell.node_mask is not None for cell in cells):
            valid_mask = torch.stack([cell.node_mask for cell in cells], dim=0).to(
                pred_orig.device
            )
        node_index, eps_k = self._select_topk_nodes(fitted, target, valid_mask)
        nodes_updated = (
            int(node_index.numel())
            if node_index is not None
            else int(pred_orig.shape[1])
        )

        loss.backward()
        if node_index is not None:
            self._mask_node_grads(node_index)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.calibrator.eval()
        if depths is not None:
            with torch.no_grad():
                served, truth = pred_orig.detach().squeeze(-1), target.squeeze(
                    -1
                )  # [K, N]
                self.memory.verify_depths(
                    truth, served, depths, diagonals=merged_diagonals
                )
                self.memory.write(
                    mem_slots, horizons, truth, served, diagonals=merged_diagonals
                )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.stats.record(
            len(cells),
            (time.perf_counter() - start_time) * 1000.0,
            merged_diagonals=merged_diagonals,
            nodes_updated=nodes_updated,
            eps_k=eps_k,
        )

    @torch.no_grad()
    def memory_correction(
        self, origin_start: int, batch_size: int
    ) -> Optional[torch.Tensor]:
        """Recalled calendar correction ``[B, H, N]`` (original units) for new rows.

        Only diagonals consumed by ``maintain`` before serving contribute, so the
        read is causal.  Returns ``None`` when the memory is disabled.
        """
        if self.memory is None:
            return None
        lag, horizon = self.calendar.input_len, self.calibrator.pred_len
        self.calendar.check_steps(
            origin_start + lag, origin_start + batch_size - 1 + lag + horizon - 1
        )
        origins = torch.arange(
            origin_start, origin_start + batch_size, device=self.device
        )
        slots = self.calendar.slots_of_rows(
            origins, self.calibrator.pred_len
        )  # [L, B, H]
        horizons = torch.arange(self.calibrator.pred_len, device=self.device).expand(
            batch_size, -1
        )
        return self.memory.served(self.memory.read_depths(slots, horizons))

    @property
    def memory_stats(self) -> dict:
        if self.memory is None:
            return dict(enabled=False)
        cfg = self.memory.config
        return dict(
            enabled=True,
            key=cfg.key,
            levels=list(self.calendar.levels),
            decay=cfg.decay,
            shrink=cfg.shrink,
            root_shrink=cfg.root_shrink,
            joint=cfg.joint,
            verify=cfg.verify,
            verify_decay=cfg.verify_decay,
            active_node_frac=round(float(self.memory.active.mean()), 4),
            depth_frac=dict(
                zip(
                    ("none",) + tuple(self.calendar.levels),
                    self.memory.depth_fractions(),
                )
            ),
            writes=self.memory.writes,
            state_floats=self.memory.state_floats,
        )

    @property
    def execution_stats(self):
        """Backend telemetry; counters distinguish graph replay from fallback."""
        if self._execution is None:
            return dict(
                requested="reference",
                reference_steps=self.stats.applied,
                graph_steps=0,
                selected_steps=0,
                capture_seconds=0.0,
                fallback_reasons={},
            )
        return self._execution.summary()

    # ------------------------------------------------- event-time feedback API
    def _require_mode(self, mode: str, api: str) -> None:
        if self.feedback_mode != mode:
            raise RuntimeError(
                f"{api}() is only available in feedback_mode={mode!r} "
                f"(current mode: {self.feedback_mode!r})"
            )

    def issue(self, origin_start: int, raw_pred: torch.Tensor) -> None:
        """Register predictions without exposing their future targets."""
        self._require_mode(EVENT_TIME_FEEDBACK, "issue")
        self._event.issue(origin_start, raw_pred)

    def observe(self, target_time: int, node_ids, values, *, arrival_time: int) -> None:
        """Accept node labels that have arrived for one target time."""
        self._require_mode(EVENT_TIME_FEEDBACK, "observe")
        self._event.observe(target_time, node_ids, values, arrival_time=arrival_time)

    def finalize(self, processing_time: int) -> None:
        """Evict remaining event-mode state; never update after the last prediction."""
        self._require_mode(EVENT_TIME_FEEDBACK, "finalize")
        self._event.finalize(processing_time)
        stats = self._event.stats
        logger.info(
            "Event Feedback Stats: arrived=%d consumed=%d duplicate=%d conflict=%d "
            "late=%d orphan=%d future_use=%d expired_node_slots=%d rounds=%d "
            "ws_groups_max=%d ws_node_cells_max=%d ws_bytes_max=%d age_mean=%.3f",
            stats.arrived,
            stats.consumed,
            stats.duplicate,
            stats.conflict,
            stats.late,
            stats.orphan,
            stats.future_use,
            stats.expired_node_slots,
            stats.rounds,
            stats.max_pending_groups,
            stats.max_node_cells,
            stats.max_bytes,
            stats.age_mean,
        )

    @property
    def event_stats(self):
        return self._event.stats if self._event is not None else None

    # MaintenanceOperator interface aliases (engine-facing names stay primary).
    def ingest(
        self, origin_start: int, raw_pred: torch.Tensor, target: torch.Tensor
    ) -> None:
        self.enqueue(origin_start, raw_pred, target)

    def maintain(self, current_origin: int) -> None:
        if self._event is not None:
            self._event.maintain(current_origin)
        else:
            self.update_available(current_origin)
