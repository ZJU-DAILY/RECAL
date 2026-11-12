"""Fused execution of fixed-replay RECAL: C1 (low-rank field) + C2 (diagonal assimilation) + C3 (calendar memory).

Same mathematics and update order as the reference path (:class:`~recal.maintenance.ForecastDiagonalAssimilator`
with :class:`~recal.calendar_memory.CalendarMemory`, legacy fixed-replay feedback, no Top-K budget), restructured
for speed.  At the tensor sizes of test-time calibration (N <= 10^4, H = 12) a tick is bound by kernel-launch and
Python overhead, not by arithmetic, so the engine removes overhead:

* feedback lives in a ring buffer of the last ``H + B`` forecast rows instead of one Python object per cell;
* the memory levels are stacked into one table: a read or a write is one gather / one scatter for all levels;
* Adam runs on one flat parameter buffer (the calibrator parameters become views into it);
* in steady state a whole tick (bs=1: maintain + serve + enqueue), one diagonal update (bs>1, every tick) or one
  merged update (bs>1, lazy k=B) is replayed as a CUDA Graph.

The default update calibrates the whole forecast rows of a diagonal (``K x N x H`` activations) like the reference
path, which is cheap at H = 12 and quadratic in H beyond it.  ``cell_wise=True`` calibrates only the ``K`` matured cells
(``K x N``): same forward values, gradients up to float rounding, so its results are not bit-exact to the reference.

The reference path stays the default and the definition of the method; this module is an opt-in execution backend.
The Adam arithmetic follows ``torch.optim.Adam`` of PyTorch 1.11 operation by operation (newer releases default to a
foreach/lerp implementation, so results there agree only up to float rounding).
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .calendar_memory import CalendarIndex, CalendarMemoryConfig
from .calibrator import masked_mae_loss, select_horizon_values


class FusedReCal:
    """Online calibrator state + feedback ring + calendar memory, stepped once per served batch.

    Args:
        calibrator: trainable calibrator ``[B, 1, N, H] -> [B, 1, N, H]`` (e.g. :class:`ReCalibrator`), or ``None``
            to remove C1 outright (no calibrator, no gradient step; the memory alone corrects the backbone).
        num_nodes, pred_len: ``N`` and ``H``.
        mean, std: normalisation of the backbone output (forecasts are de-normalised as ``y * std + mean``).
        batch_size: rows served per :meth:`step` (``B``); the last batch of a stream may be shorter.
        lr, betas, eps: Adam hyper-parameters of the calibrator.
        target_offset: RECAL target offset (cell ``(o, h)`` matures at ``o + h + target_offset``).
        update_interval: ``1`` = one update per matured diagonal; ``k > 1`` merges ``k`` diagonals (lazy).
        calendar, memory: calendar index and configuration of C3 (``None`` = memory off).
        use_graph: replay steady-state work as CUDA Graphs (requires CUDA).
        cell_wise: calibrate only the matured cells of an update; needs ``calibrator.forward_selected`` (long horizons).
    """

    def __init__(
        self,
        calibrator: Optional[nn.Module],
        *,
        num_nodes: int,
        pred_len: int,
        mean: float,
        std: float,
        batch_size: int,
        device: torch.device,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        target_offset: int = 1,
        update_interval: int = 1,
        calendar: Optional[CalendarIndex] = None,
        memory: Optional[CalendarMemoryConfig] = None,
        use_graph: bool = True,
        max_updates: int = 1 << 18,
        cell_wise: bool = False,
    ) -> None:
        if (
            cell_wise
            and calibrator is not None
            and not hasattr(calibrator, "forward_selected")
        ):
            raise ValueError("cell_wise needs a calibrator with forward_selected")
        self.cell_wise = bool(cell_wise)
        self.dev = torch.device(device)
        self.N, self.H, self.B = int(num_nodes), int(pred_len), int(batch_size)
        self.C = self.H + self.B  # ring capacity: origins still needed + one batch
        self.mean, self.std = float(mean), float(std)
        self.offset = int(target_offset)
        self.k = max(1, int(update_interval))
        if (
            self.k > self.B
        ):  # older origins would be overwritten by later rows and labels
            raise ValueError(
                "update_interval > batch_size overflows the ring of H + B origins"
            )
        self.use_graph = bool(use_graph) and self.dev.type == "cuda"
        self.ar_H = torch.arange(self.H, device=self.dev)
        self.ar_B = torch.arange(self.B, device=self.dev)
        self.hz_desc = (
            self.H - 1
        ) - self.ar_H  # horizons of a full diagonal, origins ascending
        self.raw_ring = torch.zeros(self.C, self.N, self.H, device=self.dev)
        self.lab_ring = torch.zeros(self.C, self.H, self.N, device=self.dev)
        self.next_tau = self.offset  # first target time not yet consumed
        self.enqueued = 0  # origins enqueued so far
        self.pending: List[int] = []  # lazy: matured diagonals waiting to be merged
        self.updates = 0
        self.graph_replays = 0
        self.capture_seconds = 0.0

        # ---- C1: calibrator with a flat parameter / gradient buffer and Adam state
        self.cal = calibrator
        self.params = list(calibrator.parameters()) if calibrator is not None else []
        self.lr, (self.b1, self.b2), self.eps = (
            float(lr),
            (float(betas[0]), float(betas[1])),
            float(eps),
        )
        self.adam_steps, self.k0, self.max_updates = 0, 1, int(max_updates)
        if self.params:
            total = sum(p.numel() for p in self.params)
            self.flat = torch.zeros(total, device=self.dev)
            self.flat_grad = torch.zeros(total, device=self.dev)
            pos = 0
            with torch.no_grad():
                for p in self.params:
                    n = p.numel()
                    self.flat[pos : pos + n].copy_(p.detach().reshape(-1))
                    p.data = self.flat[pos : pos + n].view_as(p)
                    p.grad = self.flat_grad[pos : pos + n].view_as(p)
                    pos += n
            self.m = torch.zeros_like(self.flat)
            self.v = torch.zeros_like(self.flat)
            # Bit-exact Adam inside a captured graph.  torch.optim.Adam (1.11) divides ``v.sqrt()`` by the CPU scalar
            # sqrt(1 - b2**k), which the GPU executes as a multiplication by float(1 / float(sqrt(.))): that factor is
            # looked up per step from a host table.  Its ``addcdiv_(value=-lr / (1 - b1**k))`` is a fused
            # multiply-add with a host scalar that no tensor op reproduces, but 1 - b1**k rounds to exactly 1.0 in
            # double from step ``k0`` on (k0 = 356 for b1 = 0.9), after which the value is the constant -lr.  Steps
            # before k0 therefore run eagerly with host scalars; graphs are used from k0 on.
            inv = [
                np.float32(1.0) / np.float32(math.sqrt(1 - self.b2**step))
                for step in range(1, self.max_updates + 1)
            ]
            self.inv_tab = torch.tensor(
                np.asarray(inv, dtype=np.float32), device=self.dev
            )
            while 1 - self.b1**self.k0 != 1.0:
                self.k0 += 1
            self.step_i = torch.zeros(1, dtype=torch.long, device=self.dev)

        # ---- C3: stacked calendar memory
        self.cfg = memory if memory is not None and memory.enabled else None
        if self.cfg is not None:
            if calendar is None:
                raise ValueError("calendar memory requires a CalendarIndex")
            if self.k > 1 and calendar.repeats_adjacent_slots():
                raise ValueError(
                    "merged diagonals (update_interval > 1) need slot_ticks = 1"
                )
            if self.k > calendar.period:
                raise ValueError(
                    "calendar memory requires update_interval <= calendar period"
                )
            sizes, rates = list(calendar.level_sizes), list(calendar.level_rates)
            self.L = len(sizes)
            offsets = [0]
            for size in sizes[:-1]:
                offsets.append(offsets[-1] + size)
            self.slot_tab = (
                calendar._slots.to(self.dev)
                + torch.tensor(offsets, device=self.dev)[:, None]
            )  # [L, num_steps] global slots
            self.P = int(calendar.input_len)
            if int(calendar.target_offset) != self.offset:
                raise ValueError("calendar and engine target offsets differ")
            self.sum_all = torch.zeros(sum(sizes), self.H, self.N, device=self.dev)
            self.cnt_all = torch.zeros_like(self.sum_all)
            root = (
                self.cfg.shrink
                if self.cfg.root_shrink is None
                else self.cfg.root_shrink
            )
            self.kappas = [float(root)] + [float(self.cfg.shrink)] * (self.L - 1)
            self.decays = [
                self.cfg.decay if r == 1 else self.cfg.decay ** (1.0 / r) for r in rates
            ]
            self.shared = [r > 1 for r in rates]
            self.dec_t = torch.tensor(self.decays, device=self.dev).view(
                -1, 1, 1
            )  # [L, 1, 1] float32
            self.select = self.cfg.select
            self.active = torch.full(
                (self.N,), 0.0 if self.cfg.verify else 1.0, device=self.dev
            )
            if self.select:
                self.verify_loss = torch.zeros(self.L + 1, self.N, device=self.dev)
                self.choice = torch.zeros(self.N, dtype=torch.long, device=self.dev)
            else:
                self.verify_gap = torch.zeros(self.N, device=self.dev)
        self.graphs: Dict[str, torch.cuda.CUDAGraph] = {}
        self.s_dev = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.tau_dev = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.raw_in = torch.zeros(self.B, 1, self.N, self.H, device=self.dev)
        self.lab_in = torch.zeros(self.B, self.H, self.N, device=self.dev)
        self.pred_out = torch.zeros(self.B, self.H, self.N, device=self.dev)
        self.cal_out = torch.zeros(self.B, 1, self.N, self.H, device=self.dev)
        self.rec_out = torch.zeros(self.B, self.H, self.N, device=self.dev)

    # ------------------------------------------------------------------------------------------ state
    def _state(self) -> List[torch.Tensor]:
        out = [self.raw_ring, self.lab_ring]
        if self.params:
            out += [self.flat, self.flat_grad, self.m, self.v, self.step_i]
        if self.cfg is not None:
            out += [self.sum_all, self.cnt_all, self.active]
            out += [self.verify_loss, self.choice] if self.select else [self.verify_gap]
        return out

    @property
    def memory_state_floats(self) -> int:
        if self.cfg is None:
            return 0
        extra = (
            self.verify_loss.numel() + self.choice.numel()
            if self.select
            else self.verify_gap.numel()
        )
        return int(self.sum_all.numel() * 2 + extra + self.active.numel())

    # ------------------------------------------------------------------------------------------ C1 + Adam
    @torch.no_grad()
    def _adam(self, graph: bool) -> None:
        """torch.optim.Adam (PyTorch 1.11 single-tensor path) on the flat buffers, operation by operation."""
        g = self.flat_grad
        self.step_i.add_(1)
        if graph:  # only replayed from step k0 on (see __init__)
            self.m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            self.v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            denom = (self.v.sqrt() * self.inv_tab[self.step_i - 1]).add_(self.eps)
            self.flat.addcdiv_(self.m, denom, value=-self.lr)
            return
        self.adam_steps += 1
        k = self.adam_steps
        bc1, bc2 = 1 - self.b1**k, 1 - self.b2**k
        self.m.mul_(self.b1).add_(g, alpha=1 - self.b1)
        self.v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
        denom = (self.v.sqrt() / math.sqrt(bc2)).add_(self.eps)
        self.flat.addcdiv_(self.m, denom, value=-(self.lr / bc1))

    # ------------------------------------------------------------------------------------------ C3
    def _usable(self, target: torch.Tensor, served: torch.Tensor) -> torch.Tensor:
        return (
            (target.abs() > self.cfg.null_tol)
            & torch.isfinite(target)
            & torch.isfinite(served)
        )

    def _read(self, slots: torch.Tensor, hz: torch.Tensor) -> torch.Tensor:
        """Back-off read-outs ``[L, *idx, N]`` (select keys) or ``[1, *idx, N]`` for global ``slots [L, *idx]``."""
        s = self.sum_all[slots, hz]
        c = self.cnt_all[slots, hz]
        values: List[torch.Tensor] = []
        for level, kappa in enumerate(self.kappas):
            values.append(
                s[level] / (c[level] + kappa)
                if not values
                else (s[level] + kappa * values[-1]) / (c[level] + kappa)
            )
        return torch.stack(values, dim=0) if self.select else values[-1].unsqueeze(0)

    def _served(self, depths: torch.Tensor) -> torch.Tensor:
        if not self.select:
            return self.active * depths[-1]
        choice = self.choice.view(*([1] * (depths.dim() - 1)), -1)
        picked = depths.gather(
            0, (choice - 1).clamp_min(0).expand_as(depths[:1])
        ).squeeze(0)
        return torch.where(choice.squeeze(0) > 0, picked, torch.zeros_like(picked))

    def _verify(
        self,
        target: torch.Tensor,
        served: torch.Tensor,
        depths: torch.Tensor,
        diagonals: int,
    ) -> None:
        if not self.cfg.verify:
            return
        valid = self._usable(target, served)
        scale = self.cfg.verify_decay ** int(diagonals)
        if self.select:
            base = target - served
            losses = torch.cat(
                [base.unsqueeze(0), base.unsqueeze(0) - depths], dim=0
            ).abs()
            losses = torch.where(valid, losses, torch.zeros_like(losses))
            mean_loss = losses.sum(1) / valid.float().sum(0).clamp_min(1.0)
            self.verify_loss.mul_(scale).add_(mean_loss)
            self.choice.copy_(torch.argmin(self.verify_loss, dim=0))
            self.active.copy_((self.choice > 0).float())
        else:
            zero = torch.zeros_like(target)
            gap = torch.where(
                valid,
                (target - served - depths[-1]).abs() - (target - served).abs(),
                zero,
            )
            mean_gap = gap.sum(0) / valid.float().sum(0).clamp_min(1.0)
            self.verify_gap.mul_(scale).add_(mean_gap)
            self.active.copy_((self.verify_gap < 0).float())

    def _write(
        self,
        slots: torch.Tensor,
        hz: torch.Tensor,
        target: torch.Tensor,
        served: torch.Tensor,
        diagonals: int,
        chunks: Optional[Sequence[slice]] = None,
        graph: bool = False,
    ) -> None:
        """``chunks``: cell ranges of the individual diagonals of a merged update (graph path, shared slot)."""
        valid = self._usable(target, served)
        residual = torch.where(valid, target - served, torch.zeros_like(target))
        weight = valid.float()
        merged = int(diagonals) > 1
        plain = [l for l in range(self.L) if not (merged and self.shared[l])]
        if len(plain) == self.L:  # one gather / scatter for all levels
            hh = hz.expand_as(slots)
            self.sum_all[slots, hh] = self.dec_t * self.sum_all[slots, hh] + residual
            self.cnt_all[slots, hh] = self.dec_t * self.cnt_all[slots, hh] + weight
            return
        for level in range(self.L):  # merged diagonals: shared (static) slot
            idx = (slots[level], hz)
            if level in plain:
                d = self.decays[level]
                self.sum_all[idx] = d * self.sum_all[idx] + residual
                self.cnt_all[idx] = d * self.cnt_all[idx] + weight
            else:
                d = self.decays[level] ** int(diagonals)
                self.sum_all[idx] = d * self.sum_all[idx]
                self.cnt_all[idx] = d * self.cnt_all[idx]
                if not graph:  # the reference operation itself
                    self.sum_all.index_put_(idx, residual, accumulate=True)
                    self.cnt_all.index_put_(idx, weight, accumulate=True)
                    continue
                # index_put_(accumulate=True) adds repeated (slot, horizon) entries in cell order (stable sort) but
                # cannot be captured on PyTorch 1.11; adding the diagonals one after another gives the same sums.
                for part in chunks:
                    sub = (slots[level][part], hz[part])
                    self.sum_all[sub] = self.sum_all[sub] + residual[part]
                    self.cnt_all[sub] = self.cnt_all[sub] + weight[part]

    # ------------------------------------------------------------------------------------------ one update
    def _update(
        self,
        raw_k: torch.Tensor,
        hz: torch.Tensor,
        target: torch.Tensor,
        slots: Optional[torch.Tensor],
        diagonals: int,
        graph: bool = False,
        chunks: Optional[Sequence[slice]] = None,
    ) -> None:
        """Reference ``ForecastDiagonalAssimilator._update_cells`` on cells ``raw_k [K, N, H]`` (``[K, N]`` when
        ``cell_wise``), ``target [K, N]``."""
        if self.cal is not None:
            self.flat_grad.zero_()
            if self.cell_wise:
                selected = self.cal.forward_selected(raw_k, hz)
            else:
                selected = select_horizon_values(
                    self.cal(raw_k.unsqueeze(1)), hz
                )  # [K, 1, N, H] -> [K, N]
            pred_orig = selected.unsqueeze(-1) * self.std + self.mean
        else:
            with torch.no_grad():
                selected = (
                    raw_k
                    if self.cell_wise
                    else select_horizon_values(raw_k.unsqueeze(1), hz)
                )
                pred_orig = selected.unsqueeze(-1) * self.std + self.mean
        depths = None
        fitted = pred_orig
        if self.cfg is not None:
            with torch.no_grad():
                depths = self._read(slots, hz)
            if self.cfg.joint and self.cal is not None:
                fitted = pred_orig + self._served(depths).unsqueeze(-1)
        if self.cal is not None:
            loss = masked_mae_loss(fitted, target.unsqueeze(-1))
            loss.backward()
            self._adam(graph)
        if self.cfg is not None:
            with torch.no_grad():
                served = pred_orig.detach().squeeze(-1)
                self._verify(target, served, depths, diagonals)
                self._write(slots, hz, target, served, diagonals, chunks, graph)

    # ------------------------------------------------------------------------------------------ cell sets
    def _raw_cells(self, ridx: torch.Tensor, hz: torch.Tensor) -> torch.Tensor:
        return self.raw_ring[ridx, :, hz] if self.cell_wise else self.raw_ring[ridx]

    def _cells_host(self, taus: Sequence[int]):
        """Cells of the given target times (eager path): ring rows, horizons, target slots, per-diagonal ranges."""
        origins, horizons, steps, chunks = [], [], [], []
        for tau in taus:
            last = tau - self.offset
            start = len(origins)
            for o in range(max(0, last - (self.H - 1)), last + 1):
                origins.append(o)
                horizons.append(last - o)
                steps.append(last + (self.P if self.cfg is not None else 0))
            chunks.append(slice(start, len(origins)))
        o = torch.tensor(origins, device=self.dev)
        hz = torch.tensor(horizons, device=self.dev)
        ridx = torch.remainder(o, self.C)
        slots = (
            self.slot_tab[:, torch.tensor(steps, device=self.dev)]
            if self.cfg is not None
            else None
        )
        return self._raw_cells(ridx, hz), hz, self.lab_ring[ridx, hz], slots, chunks

    def _cells_full_diag(self, tau: torch.Tensor):
        """A full diagonal at target time ``tau`` ([1] long tensor): every index is computed on the device."""
        o = tau - self.offset - (self.H - 1) + self.ar_H
        ridx = torch.remainder(o, self.C)
        hz = self.hz_desc
        slots = None
        if self.cfg is not None:
            slots = self.slot_tab[:, tau - self.offset + self.P].expand(self.L, self.H)
        return self._raw_cells(ridx, hz), hz, self.lab_ring[ridx, hz], slots

    def _cells_full_merged(self, s: torch.Tensor, count: Optional[int] = None):
        """``count`` (default ``B``) full diagonals ending at target time ``s``, in the reference pending order."""
        count = self.B if count is None else int(count)
        taus = s - (count - 1) + torch.arange(count, device=self.dev)  # [count]
        o = (taus[:, None] - self.offset - (self.H - 1) + self.ar_H[None, :]).reshape(
            -1
        )
        ridx = torch.remainder(o, self.C)
        hz = self.hz_desc.repeat(count)
        slots = None
        if self.cfg is not None:
            # repeat each diagonal's slots over its H cells (repeat_interleave syncs with the host on PyTorch 1.11)
            per_tau = self.slot_tab[:, taus - self.offset + self.P]  # [L, count]
            slots = (
                per_tau[:, :, None]
                .expand(self.L, count, self.H)
                .reshape(self.L, count * self.H)
            )
        return self._raw_cells(ridx, hz), hz, self.lab_ring[ridx, hz], slots

    # ------------------------------------------------------------------------------------------ serve / enqueue
    def _serve(self, s: torch.Tensor, raw: torch.Tensor):
        """Served forecast ``[B, H, N]`` plus its parts: calibrated forecast ``[B, 1, N, H]`` (normalised units)
        and the recalled correction ``[B, H, N]`` (original units, ``None`` without memory).
        """
        with torch.no_grad():
            cal = self.cal(raw) if self.cal is not None else raw
            pred = (cal[:, 0] * self.std + self.mean).permute(0, 2, 1)  # [B, H, N]
            rec = None
            if self.cfg is not None:
                rows = raw.shape[0]
                steps = (
                    (s + self.ar_B[:rows])[:, None] + self.P + self.ar_H[None, :]
                )  # [B, H]
                slots = self.slot_tab[:, steps]  # [L, B, H]
                rec = self._served(self._read(slots, self.ar_H.expand(rows, self.H)))
                pred = pred + rec
        return pred, cal, rec

    def _serve_into_buffers(self) -> None:
        pred, cal, rec = self._serve(self.s_dev, self.raw_in)
        self.pred_out.copy_(pred)
        self.cal_out.copy_(cal)
        if rec is not None:
            self.rec_out.copy_(rec)

    def _outputs(self, parts: bool):
        if not parts:
            return self.pred_out
        return self.cal_out, (self.rec_out if self.cfg is not None else None)

    def _enqueue(self, s: torch.Tensor, raw: torch.Tensor, label: torch.Tensor) -> None:
        rows = raw.shape[0]
        idx = torch.remainder(s + self.ar_B[:rows], self.C)
        self.raw_ring[idx] = raw[:, 0]
        self.lab_ring[idx] = label

    # ------------------------------------------------------------------------------------------ graphs
    def _capture(self, name: str, body) -> None:
        started = time.perf_counter()
        saved = [
            t.cpu() for t in self._state()
        ]  # on the host: a GPU copy would double the peak memory
        side = torch.cuda.Stream(device=self.dev)
        side.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(side):
            for _ in range(2):
                body()
        torch.cuda.current_stream(self.dev).wait_stream(side)
        torch.cuda.synchronize(self.dev)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        torch.cuda.synchronize(self.dev)
        with torch.no_grad():
            for t, value in zip(self._state(), saved):
                t.copy_(value)
        torch.cuda.synchronize(self.dev)
        self.graphs[name] = graph
        self.capture_seconds += time.perf_counter() - started

    def _replay(self, name: str, body) -> None:
        if name not in self.graphs:
            self._capture(name, body)
        self.graphs[name].replay()
        self.graph_replays += 1

    def _body_tick1(self) -> None:
        raw_k, hz, target, slots = self._cells_full_diag(self.s_dev)
        self._update(raw_k, hz, target, slots, 1, graph=True)
        self._serve_into_buffers()
        self._enqueue(self.s_dev, self.raw_in, self.lab_in)

    def _body_diag(self) -> None:
        raw_k, hz, target, slots = self._cells_full_diag(self.tau_dev)
        self._update(raw_k, hz, target, slots, 1, graph=True)

    def _body_merged(self) -> None:
        raw_k, hz, target, slots = self._cells_full_merged(self.s_dev)
        chunks = [slice(b * self.H, (b + 1) * self.H) for b in range(self.B)]
        self._update(raw_k, hz, target, slots, self.B, graph=True, chunks=chunks)

    def _body_serve(self) -> None:
        self._serve_into_buffers()
        self._enqueue(self.s_dev, self.raw_in, self.lab_in)

    # ------------------------------------------------------------------------------------------ public API
    def summary(self) -> Dict[str, object]:
        """Counters and memory statistics in the spirit of ``ForecastDiagonalAssimilator.memory_stats``."""
        out: Dict[str, object] = dict(
            updates=self.updates,
            adam_steps=self.adam_steps,
            graph_replays=self.graph_replays,
            graphs=sorted(self.graphs),
            capture_seconds=round(self.capture_seconds, 3),
            memory_state_floats=self.memory_state_floats,
        )
        if self.cfg is not None:
            if self.select:
                counts = (
                    torch.bincount(self.choice, minlength=self.L + 1).float() / self.N
                )
                out["depth_frac"] = [round(float(c), 4) for c in counts]
            out["active_node_frac"] = round(float(self.active.mean()), 4)
        return out

    def _full(self, tau: int) -> bool:
        return tau - self.offset - (self.H - 1) >= 0

    def _maintain_eager(self, s: int) -> None:
        for tau in range(self.next_tau, s + 1):
            last = tau - self.offset
            if last < 0 or last > self.enqueued - 1:
                continue  # no matured cell at this target time
            if self.k == 1:
                raw_k, hz, target, slots, chunks = self._cells_host([tau])
                self._update(raw_k, hz, target, slots, 1, chunks=chunks)
                self.updates += 1
            else:
                self.pending.append(tau)
                if len(self.pending) >= self.k:
                    count = len(self.pending)
                    if self._full(self.pending[0]) and self.pending == list(
                        range(self.pending[0], tau + 1)
                    ):
                        # contiguous full diagonals: build the cell set on the device (no per-cell host loop)
                        cells = self._cells_full_merged(
                            torch.full((1,), tau, dtype=torch.long, device=self.dev),
                            count,
                        )
                        raw_k, hz, target, slots = cells
                        chunks = [
                            slice(b * self.H, (b + 1) * self.H) for b in range(count)
                        ]
                    else:
                        raw_k, hz, target, slots, chunks = self._cells_host(
                            self.pending
                        )
                    self._update(raw_k, hz, target, slots, count, chunks=chunks)
                    self.updates += 1
                    self.pending = []
        self.next_tau = max(self.next_tau, s + 1)

    def step(self, s: int, raw: torch.Tensor, label: torch.Tensor, parts: bool = False):
        """Maintain (consume diagonals maturing up to ``s``), serve rows ``s..s+b-1``, enqueue their feedback.

        ``raw [b, 1, N, H]`` normalised backbone forecasts, ``label [b, H, N]`` (or ``[b, H, N, 1]``) in original
        units.  Returns the served forecasts ``[b, H, N]`` (a view of a persistent buffer when ``b == B``: copy it if
        it must outlive the next call).  With ``parts=True`` it returns ``(calibrated [b, 1, N, H] in normalised
        units, recalled [b, H, N] in original units or None)`` instead, so that a scenario can apply its own inverse
        transform exactly as its reference loop does.

        Autograd anomaly detection (enabled globally by some scenario code, e.g. STONE) only adds checks and host
        syncs that a CUDA Graph cannot capture; it is switched off within the step and restored afterwards.
        """
        with torch.autograd.set_detect_anomaly(False):
            return self._step(s, raw, label, parts)

    def _step(self, s: int, raw: torch.Tensor, label: torch.Tensor, parts: bool):
        rows = raw.shape[0]
        if label.dim() == 4:
            label = label[..., 0]
        if self.params and self.updates + s + 1 - self.next_tau + 1 > self.max_updates:
            raise RuntimeError(
                "FusedReCal: more updates than max_updates; construct it with a larger max_updates"
            )
        graph_ok = self.use_graph and rows == self.B and self.enqueued == s
        adam_ok = (
            not self.params or self.adam_steps + 1 >= self.k0
        )  # bit-exact Adam graphs only from step k0
        taus = list(range(self.next_tau, s + 1))
        if (
            graph_ok
            and adam_ok
            and self.B == 1
            and self.k == 1
            and taus == [s]
            and self._full(s)
        ):
            self.s_dev.fill_(s)
            self.raw_in.copy_(raw)
            self.lab_in.copy_(label)
            self._replay("tick1", self._body_tick1)
            self.next_tau, self.enqueued = s + 1, s + 1
            self.updates += 1
            self.adam_steps += 1 if self.params else 0
            return self._outputs(parts)
        if graph_ok and self.B > 1 and taus and all(self._full(t) for t in taus):
            if adam_ok and self.k == 1:
                for tau in taus:
                    self.tau_dev.fill_(tau)
                    self._replay("diag", self._body_diag)
                self.updates += len(taus)
                self.adam_steps += len(taus) if self.params else 0
                self.next_tau = s + 1
            elif (
                adam_ok
                and self.k == self.B
                and not self.pending
                and len(taus) == self.B
            ):
                self.s_dev.fill_(s)
                self._replay("merged", self._body_merged)
                self.updates += 1
                self.adam_steps += 1 if self.params else 0
                self.next_tau = s + 1
            else:
                self._maintain_eager(s)
            self.s_dev.fill_(s)
            self.raw_in.copy_(raw)
            self.lab_in.copy_(label)
            self._replay("serve", self._body_serve)
            self.enqueued = s + rows
            return self._outputs(parts)
        # warm-up, short last batch, non-contiguous calls or graphs disabled: eager, same arithmetic
        self._maintain_eager(s)
        s_t = torch.full((1,), s, dtype=torch.long, device=self.dev)
        pred, cal, rec = self._serve(s_t, raw)
        self._enqueue(s_t, raw, label)
        self.enqueued = s + rows
        return (cal, rec) if parts else pred
