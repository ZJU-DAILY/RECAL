"""Calendar-addressed residual memory for RECAL (optional component).

Test-time bias of a frozen spatio-temporal forecaster recurs with the calendar
clock of the *target* time (time of day, refined by day type).  A calibrator
whose parameters do not depend on the clock can only follow such a recurrent
bias by fast tracking, which also amplifies label noise.  The memory instead
stores residual statistics per calendar slot:

* write: every matured forecast-lattice diagonal shares one target time, hence
  one calendar slot per level; its ``H`` cell residuals update that slot.
* read: a new forecast row reads, for every horizon, the slot of that
  horizon's own target time (known at forecast time).

Each slot keeps a decayed residual sum and count.  The read-out shrinks a slot
toward its parent level (fine -> coarse), and the root toward zero:
``m_l = (S_l + kappa * m_{l-1}) / (C_l + kappa)`` with ``kappa > 0``.

Self-verification: since raw forecasts are cached, every matured diagonal also
yields the loss the served forecast would have had with and without the recalled
correction, both evaluated with the current calibrator.  A per-node discounted
loss difference switches the recall off where it does not help.  Under joint
(backfitting) training the calibrator of an active node no longer models the
recalled part, so the switch tests "does the recall help the current
calibrator", not "would a calibrator trained without memory do better"; a node
that is switched off is trained without recall from then on.

Resolution selection (``multi`` keys): which clock the residual recurs on is
domain dependent (traffic: time of day and day type; wind power: a persistent
level that follows the weather, not the clock).  The ``multi`` and ``anchored``
hierarchies are rooted at a *static* slot shared by all target times: it takes
one write per diagonal, and its per-write decay is rescaled so that it forgets
on the same time scale as a time-of-day slot.  Calendar slots then shrink
toward the static offset instead of toward zero.  ``anchored`` keys serve the
deepest read-out behind the usual on/off switch; ``multi`` keys let the same
counterfactual check score every depth of the back-off chain, plus "no
recall", per node, and each node serves the depth with the lowest discounted
loss.

``CalendarIndex`` is a pure key function; ``CalendarMemory`` holds the state.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

MEMORY_KEYS = (
    "off",
    "tod",
    "tod_daytype",
    "hier",
    "hier3",
    "multi",
    "multi_daytype",
    "multi_deep",
    "multi_daytype_deep",
    "multi_nostatic",
    "multi_notod",
    "anchored",
    "anchored_daytype",
)
# per-node depth selection; multi_nostatic / multi_notod are hierarchy ablations of multi_daytype
SELECT_KEYS = ("multi", "multi_daytype", "multi_nostatic", "multi_notod")
# Requestable from the command line: each "auto*" family picks its time-of-day or its
# day-type member by the weekly-structure test of the pre-test series.
AUTO_FAMILIES = {
    "auto": ("tod", "tod_daytype"),
    "auto_multi": ("multi", "multi_daytype"),
    "auto_multi_deep": ("multi_deep", "multi_daytype_deep"),
    "auto_anchored": ("anchored", "anchored_daytype"),
}
REQUEST_KEYS = MEMORY_KEYS + tuple(AUTO_FAMILIES)
_LEVELS = {
    "tod": ("tod",),
    "tod_daytype": ("tod_daytype",),
    "hier": ("tod", "tod_daytype"),
    "hier3": ("tod", "tod_daytype", "tod_dow"),
    "multi": ("static", "tod"),
    "multi_daytype": ("static", "tod", "tod_daytype"),
    "multi_deep": (
        "static",
        "tod",
    ),  # ablation: serve the deepest depth (on/off switch)
    "multi_daytype_deep": ("static", "tod", "tod_daytype"),
    "multi_nostatic": ("tod", "tod_daytype"),  # ablation: no static root
    "multi_notod": ("static", "tod_daytype"),  # ablation: no time-of-day level
    "anchored": ("static", "tod"),  # calendar slots shrink toward the static offset
    "anchored_daytype": ("static", "tod_daytype"),
}
_SLOTS_PER_PERIOD = {"static": 0, "tod": 1, "tod_daytype": 2, "tod_dow": 7}


def resolve_memory_key(requested: str, auto_key: str) -> str:
    """Concrete key for a requested one; ``auto_key`` is ``"tod"`` or ``"tod_daytype"``."""
    if requested in AUTO_FAMILIES:
        return AUTO_FAMILIES[requested][auto_key == "tod_daytype"]
    if requested not in MEMORY_KEYS:
        raise ValueError(f"unknown calendar memory key: {requested!r}")
    return requested


@dataclass(frozen=True)
class CalendarMemoryConfig:
    """Hyper-parameters of the calendar memory (``key='off'`` disables it)."""

    key: str = "off"
    decay: float = 0.9  # selected on the PatchSTG/SD validation stream
    shrink: float = 4.0
    root_shrink: Optional[float] = None
    joint: bool = True
    verify: bool = True  # per-node counterfactual self-verification switch
    verify_decay: float = 0.998  # discount per matured diagonal (selected on val)
    null_tol: float = (
        0.0  # |target| <= null_tol counts as missing (inverse-transform zeros)
    )

    @property
    def enabled(self) -> bool:
        return self.key != "off"

    @property
    def select(self) -> bool:
        """Whether nodes choose their back-off depth (``multi`` keys)."""
        return self.key in SELECT_KEYS

    def __post_init__(self) -> None:
        if self.key not in MEMORY_KEYS:
            raise ValueError(f"unknown calendar memory key: {self.key!r}")
        if self.select and not self.verify:
            raise ValueError(
                f"key {self.key!r} selects depths by verification; it needs verify=True"
            )
        if not 0.0 < self.decay <= 1.0:
            raise ValueError("decay must lie in (0, 1]")
        # kappa = 0 would turn unseen slots (and all-masked nodes) into 0/0 = NaN.
        if self.shrink <= 0 or (self.root_shrink is not None and self.root_shrink <= 0):
            raise ValueError("shrinkage pseudo-counts must be positive")
        if not 0.0 < self.verify_decay <= 1.0:
            raise ValueError("verify_decay must lie in (0, 1]")
        if self.null_tol < 0:
            raise ValueError("null_tol must be non-negative")


class CalendarIndex:
    """Map stream steps to calendar slots for each memory level.

    Args:
        tod: time-of-day slot of every stream step, ``[num_steps]`` in ``[0, period)``.
        dow: day-of-week index of every stream step, ``[num_steps]`` in ``[0, 7)``.
        period: number of time-of-day slots per day.
        key: one of ``MEMORY_KEYS`` (except ``"off"``).
        weekend_dows: ``dow`` values treated as weekend-type days.
        holiday: optional bool mask ``[num_steps]`` of steps on holidays
            (weekend-type day, weekday bucket of the last weekend day).
        input_len: backbone input length ``P``; target time ``tau`` of cell
            ``(origin, h)`` is ``origin + h + target_offset`` and lies on stream
            step ``origin + P + h``.
        target_offset: RECAL target offset.
    """

    def __init__(
        self,
        tod: torch.Tensor,
        dow: torch.Tensor,
        period: int,
        key: str,
        weekend_dows: Sequence[int] = (5, 6),
        holiday: Optional[torch.Tensor] = None,
        input_len: int = 12,
        target_offset: int = 1,
    ) -> None:
        if key not in _LEVELS:
            raise ValueError(f"calendar index needs an active key, got {key!r}")
        tod = torch.as_tensor(tod, dtype=torch.long)
        dow = torch.as_tensor(dow, dtype=torch.long).clone()
        if tod.shape != dow.shape or tod.dim() != 1:
            raise ValueError("tod and dow must be 1-D arrays of equal length")
        if int(tod.min()) < 0 or int(tod.max()) >= period:
            raise ValueError("tod values must lie in [0, period)")
        if int(dow.min()) < 0 or int(dow.max()) >= 7:
            raise ValueError("dow values must lie in [0, 7)")
        weekend = torch.zeros_like(dow, dtype=torch.bool)
        for d in weekend_dows:
            weekend |= dow == int(d)
        if holiday is not None:
            if not weekend_dows:
                raise ValueError("holidays need at least one weekend day to map onto")
            holiday = torch.as_tensor(holiday, dtype=torch.bool)
            weekend |= holiday
            dow = torch.where(holiday, torch.full_like(dow, int(weekend_dows[-1])), dow)
        keys = {
            "static": torch.zeros_like(tod),
            "tod": tod,
            "tod_daytype": tod + period * weekend.long(),
            "tod_dow": tod + period * dow,
        }
        self.period = int(period)
        self.levels: Tuple[str, ...] = _LEVELS[key]
        self.level_sizes: List[int] = [
            max(period * _SLOTS_PER_PERIOD[n], 1) for n in self.levels
        ]
        # Writes a slot receives per write of a time-of-day slot (the static slot takes every diagonal).
        self.level_rates: List[int] = [
            self.period if n == "static" else 1 for n in self.levels
        ]
        self._slots = torch.stack(
            [keys[n] for n in self.levels], dim=0
        )  # [L, num_steps]
        self.input_len = int(input_len)
        self.target_offset = int(target_offset)

    @property
    def num_steps(self) -> int:
        return self._slots.shape[1]

    def to(self, device: torch.device) -> "CalendarIndex":
        """Copy of this index whose slot table lives on ``device``."""
        moved = copy.copy(self)
        moved._slots = self._slots.to(device)
        return moved

    def check_steps(self, first: int, last: int) -> None:
        """Host-side bounds check (no device sync) for stream steps ``first..last``."""
        if first < 0 or last >= self.num_steps:
            raise IndexError(
                f"stream steps [{first}, {last}] outside the calendar index "
                f"[0, {self.num_steps - 1}]"
            )

    def repeats_adjacent_slots(self) -> bool:
        """Whether consecutive steps share a calendar slot (``slot_ticks > 1``) on a non-static level."""
        rows = [i for i, name in enumerate(self.levels) if name != "static"]
        if not rows or self.num_steps < 2:
            return False
        slots = self._slots[rows]
        return bool((slots[:, 1:] == slots[:, :-1]).any())

    def slots_of_steps(self, steps: torch.Tensor) -> torch.Tensor:
        """Slots ``[L, *steps.shape]`` of the given stream steps (callers check bounds)."""
        return self._slots[:, steps]

    def slots_of_targets(self, target_times: torch.Tensor) -> torch.Tensor:
        return self.slots_of_steps(target_times - self.target_offset + self.input_len)

    def slots_of_rows(self, origins: torch.Tensor, pred_len: int) -> torch.Tensor:
        """Slots ``[L, B, H]`` of the target times of forecast rows at ``origins``."""
        steps = (
            origins[:, None]
            + self.input_len
            + torch.arange(pred_len, device=origins.device)[None, :]
        )
        return self.slots_of_steps(steps)


class CalendarMemory:
    """Decayed residual statistics per (calendar slot, horizon, node) with back-off."""

    def __init__(
        self,
        level_sizes: Sequence[int],
        pred_len: int,
        num_nodes: int,
        config: CalendarMemoryConfig,
        device: torch.device,
        level_rates: Optional[Sequence[int]] = None,
    ) -> None:
        self.config = config
        self.pred_len = int(pred_len)
        self.sum = [
            torch.zeros(size, pred_len, num_nodes, device=device)
            for size in level_sizes
        ]
        self.cnt = [torch.zeros_like(t) for t in self.sum]
        root = config.shrink if config.root_shrink is None else config.root_shrink
        self.kappas = [float(root)] + [float(config.shrink)] * (len(level_sizes) - 1)
        if level_rates is None:
            if min(level_sizes) == 1:
                raise ValueError(
                    "a static level needs level_rates (use CalendarIndex.level_rates)"
                )
            level_rates = [1] * len(level_sizes)
        rates = [int(r) for r in level_rates]
        if len(rates) != len(level_sizes) or min(rates) < 1:
            raise ValueError("level_rates must give one positive rate per level")
        # A slot written ``rate`` times per time-of-day write decays by ``decay ** (1 / rate)``
        # per write, so the static slot and a time-of-day slot forget on the same time scale.
        self.decays = [
            config.decay if r == 1 else config.decay ** (1.0 / r) for r in rates
        ]
        self.shared = [
            r > 1 for r in rates
        ]  # merged diagonals repeat horizons on these slots
        self.writes = 0  # number of write calls (one per optimizer step)
        # Counterfactual self-verification state (per node).
        self.active = torch.full(
            (num_nodes,), 0.0 if config.verify else 1.0, device=device
        )
        if config.select:
            # Discounted loss of [no recall, depth 0, ..., depth L-1]; nodes serve the argmin.
            self.verify_loss = torch.zeros(
                len(level_sizes) + 1, num_nodes, device=device
            )
            self.choice = torch.zeros(num_nodes, dtype=torch.long, device=device)
        else:
            self.verify_gap = torch.zeros(num_nodes, device=device)

    @property
    def state_floats(self) -> int:
        stats = sum(t.numel() for t in self.sum) * 2
        if self.config.select:
            return int(
                stats
                + self.verify_loss.numel()
                + self.choice.numel()
                + self.active.numel()
            )
        return int(stats + self.verify_gap.numel() + self.active.numel())

    def valid_mask(self, target: torch.Tensor) -> torch.Tensor:
        """Supervised entries: finite and non-zero (zero-fill = missing, up to ``null_tol``)."""
        return (target.abs() > self.config.null_tol) & torch.isfinite(target)

    def _usable(self, target: torch.Tensor, served: torch.Tensor) -> torch.Tensor:
        """Supervised cells whose served (calibrated) forecast is finite as well."""
        return self.valid_mask(target) & torch.isfinite(served)

    def read(self, slots: torch.Tensor, horizons: torch.Tensor) -> torch.Tensor:
        """Back-off read-out for ``slots [L, *idx]`` and ``horizons [*idx]`` -> ``[*idx, N]``."""
        value = None
        for level, kappa in enumerate(self.kappas):
            s = self.sum[level][slots[level], horizons]
            c = self.cnt[level][slots[level], horizons]
            value = (
                s / (c + kappa) if value is None else (s + kappa * value) / (c + kappa)
            )
        return value

    def read_depths(self, slots: torch.Tensor, horizons: torch.Tensor) -> torch.Tensor:
        """Read-outs the served correction is chosen from: every back-off depth
        ``[L, *idx, N]`` for ``multi`` keys, only the deepest one ``[1, *idx, N]`` otherwise.
        """
        if not self.config.select:
            return self.read(slots, horizons).unsqueeze(0)
        values = []
        for level, kappa in enumerate(self.kappas):
            s = self.sum[level][slots[level], horizons]
            c = self.cnt[level][slots[level], horizons]
            values.append(
                s / (c + kappa)
                if not values
                else (s + kappa * values[-1]) / (c + kappa)
            )
        return torch.stack(values, dim=0)

    def served(self, depths: torch.Tensor) -> torch.Tensor:
        """Correction actually served ``[*idx, N]`` given :meth:`read_depths` output."""
        if not self.config.select:
            return self.active * depths[-1]
        choice = self.choice.view(*([1] * (depths.dim() - 1)), -1)  # [1, ..., 1, N]
        picked = depths.gather(
            0, (choice - 1).clamp_min(0).expand_as(depths[:1])
        ).squeeze(0)
        return torch.where(choice.squeeze(0) > 0, picked, torch.zeros_like(picked))

    def depth_fractions(self) -> List[float]:
        """Share of nodes serving [no recall, depth 0, ..., depth L-1]."""
        if not self.config.select:
            active = round(float(self.active.mean()), 4)
            return [round(1.0 - active, 4)] + [0.0] * (len(self.sum) - 1) + [active]
        counts = torch.bincount(self.choice, minlength=len(self.sum) + 1).float()
        return [round(float(c), 4) for c in counts / max(self.choice.numel(), 1)]

    def verify(
        self,
        target: torch.Tensor,
        served: torch.Tensor,
        recalled: torch.Tensor,
        diagonals: int = 1,
    ) -> None:
        """Update the per-node switch from matured cells ``[K, N]``.

        ``served`` is the calibrator output without recall and ``recalled`` the
        memory read-out before these cells are written; ``diagonals`` is the
        number of matured diagonals merged into this call (discount per diagonal).
        """
        if not self.config.verify:
            return
        valid = self._usable(target, served)
        zero = torch.zeros_like(target)
        gap = torch.where(
            valid, (target - served - recalled).abs() - (target - served).abs(), zero
        )
        mean_gap = gap.sum(0) / valid.float().sum(0).clamp_min(1.0)
        self.verify_gap = (
            self.config.verify_decay ** int(diagonals)
        ) * self.verify_gap + mean_gap
        self.active = (self.verify_gap < 0).float()

    def verify_depths(
        self,
        target: torch.Tensor,
        served: torch.Tensor,
        depths: torch.Tensor,
        diagonals: int = 1,
    ) -> None:
        """Counterfactual check of the read-outs ``depths [L, K, N]`` on matured cells ``[K, N]``.

        Single-depth keys keep the on/off switch of the deepest read-out (:meth:`verify`);
        ``multi`` keys score "no recall" and every depth, and each node serves the one with
        the lowest discounted loss (ties favour no recall, then the coarser depth).
        """
        if not self.config.select:
            self.verify(target, served, depths[-1], diagonals)
            return
        valid = self._usable(target, served)
        base = target - served
        losses = torch.cat(
            [base.unsqueeze(0), base.unsqueeze(0) - depths], dim=0
        ).abs()  # [L+1, K, N]
        losses = torch.where(valid, losses, torch.zeros_like(losses))
        mean_loss = losses.sum(1) / valid.float().sum(0).clamp_min(1.0)
        self.verify_loss = (
            self.config.verify_decay ** int(diagonals)
        ) * self.verify_loss + mean_loss
        self.choice = torch.argmin(self.verify_loss, dim=0)
        self.active = (self.choice > 0).float()

    def write(
        self,
        slots: torch.Tensor,
        horizons: torch.Tensor,
        target: torch.Tensor,
        served: torch.Tensor,
        diagonals: int = 1,
    ) -> None:
        """Fold residuals of K matured cells ``[K, N]`` into the state.

        Calendar levels require distinct (slot, horizon) pairs within one call; this
        holds for a single diagonal and for up to ``period`` merged diagonals when every
        step has its own time-of-day slot (the assimilator rejects merging otherwise).
        The shared static slot sees every merged diagonal: it decays once per diagonal
        and sums the residuals of repeated horizons.
        """
        valid = self._usable(target, served)
        residual = torch.where(valid, target - served, torch.zeros_like(target))
        weight = valid.float()
        for level in range(len(self.sum)):
            idx = (slots[level], horizons)
            if self.shared[level] and int(diagonals) > 1:
                d = self.decays[level] ** int(diagonals)
                self.sum[level][idx] = (
                    d * self.sum[level][idx]
                )  # repeated indices: same value
                self.cnt[level][idx] = d * self.cnt[level][idx]
                self.sum[level].index_put_(idx, residual, accumulate=True)
                self.cnt[level].index_put_(idx, weight, accumulate=True)
            else:
                d = self.decays[level]
                self.sum[level][idx] = d * self.sum[level][idx] + residual
                self.cnt[level][idx] = d * self.cnt[level][idx] + weight
        self.writes += 1
