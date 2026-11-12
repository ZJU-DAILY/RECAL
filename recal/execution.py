"""Opt-in hardware-aware execution for original, fixed-replay RECAL.

No feedback scheduling lives here. A single K=H graph holds input scratch,
autograd intermediates and Adam state on GPU; other K use selected eager updates.
The initial supported graph runtime is PyTorch 1.11 (the measured environment).
Unsupported contracts use the unmodified reference path and report the reason.
"""

import logging
import time

import torch

from .calibrator import ReCalibrator, masked_mae_loss, select_raw_horizon_values

logger = logging.getLogger(__name__)
EXECUTION_BACKENDS = ("reference", "hardware_aware")


class GraphUpdate:
    """One bounded graph for K=H; changing K never merges/reorders updates.

    Warmup/capture restore parameters and Adam moments before any real update.
    Graph outputs/gradients are persistent scratch, valid until the next replay.
    The caller's Adam state stays usable by the reference optimizer on fallback.
    """

    def __init__(self, model, optimizer, mean, std, k, optimizer_in_graph=True):
        if torch.is_anomaly_enabled():
            raise ValueError(
                "CUDA graph capture requires anomaly detection disabled; use selected for debugging"
            )
        self.model, self.optimizer = model, optimizer
        self.mean, self.std, self.k = mean, std, k
        self.optimizer_in_graph = optimizer_in_graph
        if len(optimizer.param_groups) != 1 or type(optimizer) is not torch.optim.Adam:
            raise ValueError("graph prototype requires a single torch.optim.Adam group")
        group = optimizer.param_groups[0]
        if group["weight_decay"] or group["amsgrad"] or group.get("maximize", False):
            raise ValueError("unsupported Adam variant")
        self.params = list(model.parameters())
        if [id(p) for p in group["params"]] != [id(p) for p in self.params]:
            raise ValueError("graph requires optimizer parameters in calibrator order")
        if any(p.dtype != torch.float32 or not p.is_cuda for p in self.params):
            raise ValueError("graph prototype requires CUDA float32 parameters")
        self.group = group
        self.raw = torch.zeros(k, model.num_nodes, device=self.params[0].device)
        self.hz = torch.arange(k, device=self.raw.device) % model.pred_len
        self.target = torch.ones(k, model.num_nodes, 1, device=self.raw.device)
        self.step = torch.zeros((), device=self.raw.device, dtype=torch.float64)
        self.beta1 = torch.tensor(
            group["betas"][0], device=self.raw.device, dtype=torch.float64
        )
        self.beta2 = torch.tensor(
            group["betas"][1], device=self.raw.device, dtype=torch.float64
        )
        self.capture_seconds = 0.0
        self._capture()

    @torch.no_grad()
    def _adam(self):
        # Match 1.11's separate mul/add, addcmul, sqrt/div/add operations.
        self.step.add_(1)
        beta1, beta2 = self.group["betas"]
        bc1 = 1 - self.beta1**self.step
        bc2sqrt = (1 - self.beta2**self.step).sqrt().float()
        step_size = (-self.group["lr"] / bc1).float()
        for p in self.params:
            s = self.optimizer.state[p]
            m, v = s["exp_avg"], s["exp_avg_sq"]
            m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
            denom = (v.sqrt() / bc2sqrt).add_(self.group["eps"])
            p.add_((m / denom) * step_size)

    def _body(self):
        for p in self.params:
            p.grad = None
        self.output = self.model.forward_selected(self.raw, self.hz)
        self.loss = masked_mae_loss(
            self.output.unsqueeze(-1) * self.std + self.mean, self.target
        )
        self.loss.backward()
        if self.optimizer_in_graph:
            self._adam()

    def _capture(self):
        started = time.perf_counter()
        # Production 1.11 uses integer step; newer runtimes use CPU scalar.
        modern = int(torch.__version__.split(".")[0]) >= 2
        with torch.no_grad():
            for p in self.params:
                s = self.optimizer.state[p]
                if not s:
                    s.update(
                        step=torch.tensor(0.0) if modern else 0,
                        exp_avg=torch.zeros_like(p),
                        exp_avg_sq=torch.zeros_like(p),
                    )
            saved = [
                (
                    p.clone(),
                    self.optimizer.state[p]["exp_avg"].clone(),
                    self.optimizer.state[p]["exp_avg_sq"].clone(),
                )
                for p in self.params
            ]
        stream = torch.cuda.Stream(device=self.raw.device)
        stream.wait_stream(torch.cuda.current_stream(self.raw.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._body()
        torch.cuda.current_stream(self.raw.device).wait_stream(stream)
        torch.cuda.synchronize(self.raw.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._body()
        self.grads = [p.grad for p in self.params]
        with torch.no_grad():
            for p, (value, m, v) in zip(self.params, saved):
                p.copy_(value)
                self.optimizer.state[p]["exp_avg"].copy_(m)
                self.optimizer.state[p]["exp_avg_sq"].copy_(v)
                p.grad = None
        torch.cuda.synchronize(self.raw.device)
        self.capture_seconds = time.perf_counter() - started

    def run(self, raw, hz, target):
        if raw.shape != self.raw.shape:
            raise ValueError("graph shape mismatch")
        self.raw.copy_(raw)
        self.hz.copy_(hz)
        self.target.copy_(target)
        if self.optimizer_in_graph:
            steps = [int(self.optimizer.state[p]["step"]) for p in self.params]
            if len(set(steps)) != 1:
                raise ValueError("graph requires equal Adam step counters")
            self.step.fill_(steps[0])
        self.graph.replay()
        for p, grad in zip(self.params, self.grads):
            p.grad = grad
            if self.optimizer_in_graph:
                self.optimizer.state[p]["step"] += 1
        if not self.optimizer_in_graph:
            self.optimizer.step()
        return self.output, self.loss


class HardwareAwareUpdate:
    """Execution adapter only; the assimilator owns diagonal order and stats."""

    def __init__(self, assimilator):
        self.assim = assimilator
        self.graph_update = None
        self.signature = None
        self.capture_seconds = 0.0
        self.graph_steps = 0
        self.selected_steps = 0
        self.reference_steps = 0
        self.fallback_reasons = {}
        self._announced = False

    def _unsupported(self, cells):
        a = self.assim
        m, opt = a.calibrator, a.optimizer
        if a.feedback_mode != "legacy":
            return "optional_event_time_uses_reference"
        if a.maintenance_budget:
            return "optional_topk_uses_reference"
        if type(m) is not ReCalibrator or not (
            m.use_horizon_domain and m.use_node_domain
        ):
            return "requires_dual_domain_ReCalibrator"
        if m._forward_hooks or m._forward_pre_hooks or m._backward_hooks:
            return "module_hooks_require_reference"
        params = list(m.parameters())
        if any(p._backward_hooks for p in params):
            return "gradient_hooks_require_reference"
        if a.device.type != "cuda":
            return "requires_cuda_fp32_trainable_parameters"
        device_index = (
            a.device.index
            if a.device.index is not None
            else torch.cuda.current_device()
        )
        if any(
            not p.is_cuda
            or p.device.index != device_index
            or p.dtype != torch.float32
            or not p.requires_grad
            for p in params
        ):
            return "requires_cuda_fp32_trainable_parameters"
        if any(
            c.raw_pred.shape != (1, 1, m.num_nodes, m.pred_len)
            or c.target.shape != (m.num_nodes, 1)
            or c.raw_pred.dtype != torch.float32
            or c.target.dtype != torch.float32
            for c in cells
        ):
            return "unsupported_cell_shape_or_dtype"
        if not torch.__version__.startswith("1.11."):
            return "unvalidated_torch_version"
        if type(opt) is not torch.optim.Adam or len(opt.param_groups) != 1:
            return "requires_single_group_Adam"
        g = opt.param_groups[0]
        if g["weight_decay"] or g["amsgrad"] or g.get("maximize", False):
            return "unsupported_Adam_variant"
        if [id(p) for p in g["params"]] != [id(p) for p in params]:
            return "optimizer_parameter_order"
        if any(isinstance(g[key], torch.Tensor) for key in ("lr", "eps")):
            return "requires_scalar_hyperparameters"
        if any(p.grad is not None for p in params):
            return "pending_gradients"
        steps = [int(opt.state.get(p, {}).get("step", 0)) for p in params]
        if len(set(steps)) != 1:
            return "unequal_Adam_steps"
        return None

    def _signature(self):
        a = self.assim
        g = a.optimizer.param_groups[0]
        params = list(a.calibrator.parameters())
        # Recapture after hyperparameter changes, state loading or tensor moves.
        return (
            id(a.optimizer),
            a.mean,
            a.std,
            g["lr"],
            g["eps"],
            g["betas"],
            tuple(p.data_ptr() for p in params),
            tuple(
                tuple(
                    (
                        a.optimizer.state.get(p, {}).get(key).data_ptr()
                        if key in a.optimizer.state.get(p, {})
                        else None
                    )
                    for key in ("exp_avg", "exp_avg_sq")
                )
                for p in params
            ),
        )

    def run(self, cells):
        reason = self._unsupported(cells)
        if reason:
            self.reference_steps += 1
            self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1
            if self.fallback_reasons[reason] == 1:
                logger.warning("RECAL hardware_aware fallback to reference: %s", reason)
            return False
        a = self.assim
        if not self._announced:
            logger.info(
                "RECAL execution=hardware_aware feedback=legacy; "
                "selected horizons, persistent CUDA Graph/Adam; "
                "anomaly checks disabled only within this update"
            )
            self._announced = True
        # Bulk raw packing intentionally trades traffic for fewer Python slices.
        # No full calibrated [K,N,H] activation is created.
        raw = torch.cat([c.raw_pred for c in cells], dim=0).to(a.device)
        hz = torch.tensor([c.horizon for c in cells], device=a.device, dtype=torch.long)
        target = torch.stack([c.target for c in cells], dim=0).to(a.device)
        raw = select_raw_horizon_values(raw, hz)
        a.calibrator.train()
        with torch.autograd.set_detect_anomaly(False):
            if len(cells) == a.calibrator.pred_len:
                signature = self._signature()
                if self.graph_update is None or signature != self.signature:
                    self.graph_update = None
                    self.graph_update = GraphUpdate(
                        a.calibrator, a.optimizer, a.mean, a.std, len(cells)
                    )
                    self.capture_seconds += self.graph_update.capture_seconds
                    self.signature = self._signature()
                self.graph_update.run(raw, hz, target)
                self.graph_steps += 1
            else:
                output = a.calibrator.forward_selected(raw, hz)
                loss = masked_mae_loss(output.unsqueeze(-1) * a.std + a.mean, target)
                loss.backward()
                a.optimizer.step()
                self.selected_steps += 1
        a.optimizer.zero_grad(set_to_none=True)
        a.calibrator.eval()
        return True

    def summary(self):
        return dict(
            requested="hardware_aware",
            graph_steps=self.graph_steps,
            selected_steps=self.selected_steps,
            reference_steps=self.reference_steps,
            capture_seconds=self.capture_seconds,
            fallback_reasons=dict(self.fallback_reasons),
        )
