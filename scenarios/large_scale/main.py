import math
import time
import torch
import torch.nn as nn
import random
import argparse
import numpy as np
import configparser
import os
import sys
from collections import deque
from tqdm import tqdm

REPO_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from recal import (
    REQUEST_KEYS,
    CalendarMemoryConfig,
    build_calendar_index,
    per_step_series,
    resolve_memory_key,
    select_calendar_key,
    verify_decay_from_halflife,
    ReCalibrator,
    EVENT_TIME_FEEDBACK,
    FEEDBACK_MODES,
    ForecastDiagonalAssimilator,
    LEGACY_FEEDBACK,
    StreamingTraceFeeder,
    count_trainable_params,
)
from recal.fused_engine import FusedReCal
from models.model import PatchSTG
from models.mage import MAGEWrapper
from models.sqlinear import SqLinear
from lib.utils import log_string, loadData, _compute_loss, metric
from lib.sqlinear_utils import loadData as load_sqlinear_data

TEST_MODES = ("recal",)
BACKBONES = ("patchstg", "mage", "sqlinear")
RUN_MODES = ("train", "test", "both")


def normalize_test_mode(mode: str) -> str:
    if mode not in TEST_MODES:
        raise ValueError(f"Unsupported test_mode: {mode}. Use one of {TEST_MODES}.")
    return mode


class Solver(object):
    DEFAULTS = {}

    @staticmethod
    def _resolve_device(cuda_id):
        if str(cuda_id) == "-1" or not torch.cuda.is_available():
            return torch.device("cpu")
        try:
            device_idx = int(cuda_id)
        except (TypeError, ValueError):
            log_string(log, f'Invalid cuda id "{cuda_id}", fallback to cuda:0')
            device_idx = 0
        device_count = torch.cuda.device_count()
        if device_idx < 0 or device_idx >= device_count:
            log_string(
                log,
                f"cuda:{device_idx} is unavailable on this machine, fallback to cuda:0",
            )
            device_idx = 0
        return torch.device(f"cuda:{device_idx}")

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.backbone = getattr(self, "backbone", "patchstg").lower()
        if self.backbone not in BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {self.backbone}. Use one of {BACKBONES}."
            )
        self.feedback_delay = max(0, int(self.feedback_delay))
        log_string(log, "\n------------ Loading Data -------------")
        if self.backbone == "sqlinear":
            (
                self.trainX,
                self.trainY,
                self.trainXTE,
                self.trainYTE,
                self.valX,
                self.valY,
                self.valXTE,
                self.valYTE,
                self.testX,
                self.testY,
                self.testXTE,
                self.testYTE,
                self.mean,
                self.std,
                self.ori_parts_idx,
                self.reo_parts_idx,
                self.reo_all_idx,
            ) = load_sqlinear_data(
                self.traffic_file,
                self.meta_file,
                self.input_len,
                self.output_len,
                self.train_ratio,
                self.test_ratio,
                self.adj_file,
                self.tod,
                self.dow,
                self.spa_patchsize,
                log,
            )
        else:
            (
                self.trainX,
                self.trainY,
                self.trainXTE,
                self.trainYTE,
                self.valX,
                self.valY,
                self.valXTE,
                self.valYTE,
                self.testX,
                self.testY,
                self.testXTE,
                self.testYTE,
                self.mean,
                self.std,
                self.ori_parts_idx,
                self.reo_parts_idx,
                self.reo_all_idx,
            ) = loadData(
                self.traffic_file,
                self.meta_file,
                self.input_len,
                self.output_len,
                self.train_ratio,
                self.test_ratio,
                self.adj_file,
                self.recur_times,
                self.tod,
                self.dow,
                self.spa_patchsize,
                log,
            )
        log_string(log, "------------ End -------------\n")
        self.best_epoch = 0
        self.device = self._resolve_device(self.cuda)
        log_string(log, f"Using device: {self.device}")
        if self.device.type == "cuda":
            log_string(
                log, f"CUDA device name: {torch.cuda.get_device_name(self.device)}"
            )
        self.build_model()

    def build_model(self):
        if self.backbone == "patchstg":
            self.model = PatchSTG(
                self.tem_patchsize,
                self.tem_patchnum,
                self.node_num,
                self.spa_patchsize,
                self.spa_patchnum,
                self.tod,
                self.dow,
                self.layers,
                self.factors,
                self.input_dims,
                self.node_dims,
                self.tod_dims,
                self.dow_dims,
                self.ori_parts_idx,
                self.reo_parts_idx,
                self.reo_all_idx,
            ).to(self.device)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            lr_milestones = [1, 35, 40]
        elif self.backbone == "mage":
            self.model = MAGEWrapper(
                node_num=self.node_num,
                seq_len=self.input_len,
                horizon=self.output_len,
                tod=self.tod,
                dow=self.dow,
                model_dim=self.mage_model_dim,
                recur_num=self.mage_recur_num,
                topk=self.mage_topk,
                graph_gen_dim=self.mage_graph_gen_dim,
                ffn_dropout=self.mage_ffn_dropout,
            ).to(self.device)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            lr_milestones = [1, 30, 50, 70, 90, 130, 170, 210, 250, 290, 300]
        elif self.backbone == "sqlinear":
            if self.tem_patchsize * self.tem_patchnum != self.input_len:
                raise ValueError(
                    f"SqLinear temporal patch mismatch: {self.tem_patchsize} * {self.tem_patchnum} != input_len {self.input_len}"
                )
            if self.trainX.shape[2] != self.node_num:
                raise ValueError(
                    f"SqLinear node count mismatch: config {self.node_num} vs data {self.trainX.shape[2]}"
                )
            if self.spa_patchnum * self.spa_patchsize != len(self.reo_all_idx):
                raise ValueError(
                    f"SqLinear spatial patch mismatch: {self.spa_patchnum} * {self.spa_patchsize} != {len(self.reo_all_idx)} reordered nodes"
                )
            self.model = SqLinear(
                self.tem_patchsize,
                self.tem_patchnum,
                self.node_num,
                self.spa_patchsize,
                self.spa_patchnum,
                self.tod,
                self.dow,
                self.layers,
                self.input_dims,
                self.node_dims,
                self.tod_dims,
                self.dow_dims,
                self.ori_parts_idx,
                self.reo_parts_idx,
                self.reo_all_idx,
                output_len=self.output_len,
                rank=self.sqlinear_node_rank,
                mlp_ratio=self.sqlinear_mlp_ratio,
            ).to(self.device)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            lr_milestones = [30, 60]
        log_string(log, f"Using backbone: {self.backbone}")
        log_string(
            log,
            f"Model parameters: {sum((p.numel() for p in self.model.parameters())):,}",
        )
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=lr_milestones, gamma=0.5
        )

    def _to_te_tensor(self, value: np.ndarray) -> torch.Tensor:
        value = np.ascontiguousarray(value)
        if not value.flags.writeable:
            value = value.copy()
        return torch.from_numpy(value).to(self.device)

    def _to_norm_x_tensor(self, value: np.ndarray) -> torch.Tensor:
        normalized = np.ascontiguousarray(
            (value - self.mean) / self.std, dtype=np.float32
        )
        return torch.from_numpy(normalized).to(self.device)

    def _to_label_tensor(self, value: np.ndarray) -> torch.Tensor:
        value = np.ascontiguousarray(value, dtype=np.float32)
        if not value.flags.writeable:
            value = value.copy()
        return torch.from_numpy(value).to(self.device)

    def _save_model_state(self) -> None:
        model_dir = os.path.dirname(self.model_file)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_file)

    def _load_model_state(self) -> str:
        load_path = self.model_file
        if not os.path.exists(load_path):
            legacy_path = getattr(self, "legacy_model_file", "")
            if legacy_path and os.path.exists(legacy_path):
                log_string(
                    log,
                    f"Model file `{load_path}` not found; loading legacy checkpoint `{legacy_path}`.",
                )
                load_path = legacy_path
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"Model checkpoint not found: `{self.model_file}`. legacy_model_file=`{getattr(self, 'legacy_model_file', '')}`"
            )
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        return load_path

    def _log_efficiency_stats(
        self,
        method_name: str,
        n_params: int,
        peak_mem_mb: float,
        samples_per_sec: float,
        adapt_ms: float,
        infer_ms: float,
        wall=None,
    ) -> None:
        """打印效率指标汇总表 (Efficiency Metrics Summary)

        ``wall``: optional list of ``(ms, samples)`` per batch, timed around the whole loop body
        (serving, adaptation and bookkeeping, synchronised).  The first 10% of the batches are
        treated as warm-up for the wall-clock lines.
        """
        sep = "=" * 68
        log_string(log, "\n" + sep)
        log_string(log, "  EFFICIENCY METRICS SUMMARY")
        log_string(log, sep)
        log_string(log, f"  {'Method':<40} {method_name}")
        log_string(log, f"  {'# Params (calibrator) ↓':<40} {n_params:,}")
        log_string(log, f"  {'Peak mem (MB) ↓':<40} {peak_mem_mb:.2f}")
        log_string(log, f"  {'Samples/sec ↑':<40} {samples_per_sec:.1f}")
        log_string(log, f"  {'Adaptation time/batch (ms) ↓':<40} {adapt_ms:.3f}")
        log_string(log, f"  {'Inference time/batch (ms) ↓':<40} {infer_ms:.3f}")
        if wall:
            tail = wall[max(1, len(wall) // 10) :] or wall
            wall_ms = [ms for ms, _ in tail]
            tail_sec = sum(wall_ms) / 1000.0
            log_string(
                log,
                f"  {'Wall time/batch, median (ms) ↓':<40} {float(np.median(wall_ms)):.3f}",
            )
            log_string(
                log,
                f"  {'Wall samples/sec (after warm-up) ↑':<40} {(sum((n for _, n in tail)) / tail_sec if tail_sec > 0 else 0.0):.1f}",
            )
        log_string(log, sep + "\n")

    def vali(self):
        self.model.eval()
        num_val = self.valX.shape[0]
        pred = []
        label = []
        num_batch = math.ceil(num_val / self.batch_size)
        with torch.no_grad():
            for batch_idx in range(num_batch):
                if isinstance(self.model, torch.nn.Module):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(num_val, (batch_idx + 1) * self.batch_size)
                    X = self.valX[start_idx:end_idx]
                    Y = self.valY[start_idx:end_idx]
                    TE = self._to_te_tensor(self.valXTE[start_idx:end_idx])
                    NormX = self._to_norm_x_tensor(X)
                    y_hat = self.model(NormX, TE)
                    pred.append(y_hat.cpu().numpy() * self.std + self.mean)
                    label.append(Y)
        pred = np.concatenate(pred, axis=0)
        label = np.concatenate(label, axis=0)
        maes = []
        rmses = []
        mapes = []
        for i in range(pred.shape[1]):
            mae, rmse, mape = metric(pred[:, i, :], label[:, i, :])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(
                log,
                "step %d, mae: %.4f, rmse: %.4f, mape: %.4f" % (i + 1, mae, rmse, mape),
            )
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(
            log, "average, mae: %.4f, rmse: %.4f, mape: %.4f" % (mae, rmse, mape)
        )
        return (np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0))

    def train(self):
        if self.backbone == "sqlinear":
            raise NotImplementedError(
                "SqLinear is migrated for test-time modes only (frozen checkpoints); train it with the original SqLinear repository (AMP + ReduceLROnPlateau)."
            )
        log_string(log, "======================TRAIN MODE======================")
        min_loss = 10000000.0
        num_train = self.trainX.shape[0]
        for epoch in tqdm(range(1, self.max_epoch + 1)):
            self.model.train()
            train_l_sum, train_acc_sum, batch_count, start = (0.0, 0.0, 0, time.time())
            permutation = np.random.permutation(num_train)
            num_batch = math.ceil(num_train / self.batch_size)
            with tqdm(total=num_batch) as pbar:
                for batch_idx in range(num_batch):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(num_train, (batch_idx + 1) * self.batch_size)
                    batch_index = permutation[start_idx:end_idx]
                    X = self.trainX[batch_index]
                    Y = self.trainY[batch_index]
                    TE = self._to_te_tensor(self.trainXTE[batch_index])
                    NormX = self._to_norm_x_tensor(X)
                    Y = self._to_label_tensor(Y)
                    self.optimizer.zero_grad()
                    y_hat = self.model(NormX, TE)
                    loss = _compute_loss(Y, y_hat * self.std + self.mean)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.optimizer.step()
                    train_l_sum += loss.cpu().item()
                    batch_count += 1
                    pbar.update(1)
            log_string(
                log,
                "epoch %d, lr %.6f, loss %.4f, time %.1f sec"
                % (
                    epoch,
                    self.optimizer.param_groups[0]["lr"],
                    train_l_sum / batch_count,
                    time.time() - start,
                ),
            )
            mae, rmse, mape = self.vali()
            self.lr_scheduler.step()
            if mae[-1] < min_loss:
                self.best_epoch = epoch
                min_loss = mae[-1]
                self._save_model_state()
        log_string(log, f"Best epoch is: {self.best_epoch}")

    def probe_bias(self, window=200):
        import numpy as _np

        log_string(log, "==================== BIAS PROBE MODE ====================")
        self._load_model_state()
        self.model.eval()
        num_val = self.testX.shape[0]
        num_batch = math.ceil(num_val / self.batch_size)
        preds, labels = ([], [])
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batch)):
                if not isinstance(self.model, torch.nn.Module):
                    continue
                start_idx = batch_idx * self.batch_size
                end_idx = min(num_val, (batch_idx + 1) * self.batch_size)
                X = self.testX[start_idx:end_idx]
                Y = self.testY[start_idx:end_idx]
                TE = self._to_te_tensor(self.testXTE[start_idx:end_idx])
                NormX = self._to_norm_x_tensor(X)
                y_hat = self.model(NormX, TE)
                preds.append(y_hat.cpu().numpy() * self.std + self.mean)
                labels.append(Y)
        pred = _np.concatenate(preds, axis=0)
        label = _np.concatenate(labels, axis=0)

        def _to_shn(a):
            a = _np.asarray(a)
            if a.ndim == 4:
                a = a[..., 0]
            return a

        pred = _to_shn(pred)
        label = _to_shn(label)
        diff = (pred - label).astype(_np.float64)
        S, H, N = diff.shape
        bias = diff.mean(axis=0)
        sv = _np.linalg.svd(bias, compute_uv=False)
        energy = float((sv**2).sum()) + 1e-12
        fro = float(_np.sqrt((bias**2).sum()))
        caps = []
        for r in (1, 2, 4, 8):
            if r <= len(sv):
                caps.append("top%d=%.4f" % (r, float((sv[:r] ** 2).sum()) / energy))
        log_string(
            log,
            "BiasField shape=%dx%d, ||B||_F=%.4f, mean|bias|=%.4f"
            % (H, N, fro, float(_np.abs(bias).mean())),
        )
        log_string(
            log,
            "BiasField energy captured: %s (sv: %s)"
            % (", ".join(caps), ", ".join(("%.3f" % v for v in sv[:8]))),
        )
        wb = []
        i = 0
        while i < S:
            wb.append(diff[i : i + window].mean(axis=0))
            i += window
        W = len(wb)
        gb = _np.mean(_np.stack(wb, 0), axis=0).ravel() if W else None
        drifts, cosw = ([], [])
        for t in range(1, W):
            a, b = (wb[t - 1], wb[t])
            drifts.append(
                float(_np.sqrt(((b - a) ** 2).sum()) / (_np.sqrt((b**2).sum()) + 1e-12))
            )
        for t in range(W):
            bt = wb[t].ravel()
            cosw.append(
                float(bt @ gb / (_np.linalg.norm(bt) * _np.linalg.norm(gb) + 1e-12))
            )
        ph = _np.abs(diff).mean(axis=(0, 2))
        log_string(
            log,
            "DynProbe windows=%d (size~%d), avg_consecutive_drift=%.4f, avg_cos_to_global=%.4f"
            % (
                W,
                window,
                float(_np.mean(drifts)) if drifts else 0.0,
                float(_np.mean(cosw)) if cosw else 1.0,
            ),
        )
        log_string(
            log,
            "DynProbe per-horizon mean|bias|: %s"
            % ", ".join(("h%d=%.3f" % (i + 1, ph[i]) for i in range(H))),
        )

    def test(self):
        log_string(log, "======================TEST MODE======================")
        self._load_model_state()
        self.model.eval()
        num_val = self.testX.shape[0]
        pred = []
        label = []
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        _infer_times: list = []
        _wall: list = []
        _total_samples = 0
        num_batch = math.ceil(num_val / self.batch_size)
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batch)):
                if isinstance(self.model, torch.nn.Module):
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    _tw = time.perf_counter()
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(num_val, (batch_idx + 1) * self.batch_size)
                    X = self.testX[start_idx:end_idx]
                    Y = self.testY[start_idx:end_idx]
                    TE = self._to_te_tensor(self.testXTE[start_idx:end_idx])
                    NormX = self._to_norm_x_tensor(X)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    _t0 = time.perf_counter()
                    y_hat = self.model(NormX, TE)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    _infer_times.append((time.perf_counter() - _t0) * 1000)
                    _total_samples += end_idx - start_idx
                    pred.append(y_hat.cpu().numpy() * self.std + self.mean)
                    label.append(Y)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    _wall.append(
                        ((time.perf_counter() - _tw) * 1000, end_idx - start_idx)
                    )
        pred = np.concatenate(pred, axis=0)
        label = np.concatenate(label, axis=0)
        maes = []
        rmses = []
        mapes = []
        for i in range(pred.shape[1]):
            mae, rmse, mape = metric(pred[:, i, :], label[:, i, :])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(
                log,
                "step %d, mae: %.4f, rmse: %.4f, mape: %.4f" % (i + 1, mae, rmse, mape),
            )
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(
            log, "average, mae: %.4f, rmse: %.4f, mape: %.4f" % (mae, rmse, mape)
        )
        _peak_mem = (
            torch.cuda.max_memory_allocated(self.device) / 1024**2
            if self.device.type == "cuda"
            else 0.0
        )
        _avg_infer = float(np.mean(_infer_times)) if _infer_times else 0.0
        _total_infer_sec = sum(_infer_times) / 1000.0
        _samples_sec = (
            _total_samples / _total_infer_sec if _total_infer_sec > 0 else 0.0
        )
        self._log_efficiency_stats(
            method_name="Baseline (no calibration)",
            n_params=0,
            peak_mem_mb=_peak_mem,
            samples_per_sec=_samples_sec,
            adapt_ms=0.0,
            infer_ms=_avg_infer,
            wall=_wall,
        )
        return (np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0))

    def _test_calendar_index(self, requested: str, target_offset: int):
        """Calendar index of the test stream; key/weekend decided on the TRAIN split only.

        Mirrors ``loadData``: TE[i] = (i % tod, (i // tod) % dow) over the whole series.
        """
        P, Q, tpd = (self.input_len, self.output_len, int(self.tod))
        if P != Q:
            raise ValueError(
                "calendar memory reconstructs splits from seq2instance windows, which assumes input_len == output_len"
            )
        n_train = self.trainX.shape[0] + P + Q - 1
        n_val = self.valX.shape[0] + P + Q - 1
        n_test = self.testX.shape[0] + P + Q - 1
        steps = np.arange(n_train + n_val + n_test)
        tod, dow = (steps % tpd, steps // tpd % int(self.dow))
        null_tol = float(getattr(self, "recal_memory_null_tol", 0.0))
        train_series = np.concatenate(
            [
                per_step_series(part, null_tol)
                for part in (
                    self.trainX[:, 0, :, 0],
                    self.trainX[-1, 1:, :, 0],
                    self.trainY[-1, :, :, 0],
                )
            ]
        )
        override = str(getattr(self, "recal_memory_weekend", "") or "")
        auto_key, weekend, score = select_calendar_key(
            train_series,
            tod[:n_train],
            dow[:n_train],
            tpd,
            weekend_override=(
                tuple((int(d) for d in override.split(","))) if override else None
            ),
        )
        key = resolve_memory_key(requested, auto_key)
        slot_ticks = int(getattr(self, "recal_memory_slot_ticks", 1))
        log_string(
            log,
            "RECAL calendar: key=%s (requested %s), weekend_dows=%s, weekly_score=%.2f, slot_ticks=%d"
            % (key, requested, weekend, score, slot_ticks),
        )
        calendar = build_calendar_index(
            tod,
            dow,
            tpd,
            key,
            first_input_step=n_train + n_val,
            num_samples=self.testX.shape[0],
            seq_len=P,
            horizon=Q,
            weekend_dows=weekend,
            slot_ticks=slot_ticks,
            target_offset=target_offset,
        )
        return (calendar, key)

    def test_with_recal(self):
        log_string(log, "==================RECAL TEST==================")
        self._load_model_state()
        self.model.eval()
        recal_rank = int(getattr(self, "recal_rank", 4))
        recal_lr = float(getattr(self, "recal_lr", 0.0001))
        recal_target_offset = int(getattr(self, "recal_target_offset", 1))
        recal_max_batches = int(getattr(self, "recal_max_batches", 0))
        recal_update_interval = int(getattr(self, "recal_update_interval", 1))
        recal_maintenance_budget = int(getattr(self, "recal_maintenance_budget", 0))
        recal_feedback_mode = str(getattr(self, "recal_feedback_mode", LEGACY_FEEDBACK))
        recal_allowed_lateness = int(getattr(self, "recal_allowed_lateness", 0))
        recal_feedback_trace = getattr(self, "recal_feedback_trace", None)
        requested_key = str(getattr(self, "recal_memory", "off"))
        calendar, memory_key = (None, "off")
        if requested_key != "off":
            calendar, memory_key = self._test_calendar_index(
                requested_key, recal_target_offset
            )
        root_shrink = getattr(self, "recal_memory_root_shrink", None)
        halflife = getattr(self, "recal_memory_verify_halflife_days", None)
        verify_decay = (
            verify_decay_from_halflife(float(halflife), int(self.tod))
            if halflife is not None
            else float(getattr(self, "recal_memory_verify_decay", 0.998))
        )
        memory_cfg = CalendarMemoryConfig(
            key=memory_key,
            decay=float(getattr(self, "recal_memory_decay", 0.9)),
            shrink=float(getattr(self, "recal_memory_shrink", 4.0)),
            root_shrink=None if root_shrink is None else float(root_shrink),
            joint=bool(int(getattr(self, "recal_memory_joint", 1))),
            verify=bool(int(getattr(self, "recal_memory_verify", 1))),
            verify_decay=verify_decay,
            null_tol=float(getattr(self, "recal_memory_null_tol", 0.0)),
        )
        calibrator = ReCalibrator(
            self.model.node_num,
            self.output_len,
            recal_rank,
            use_horizon_domain=True,
            use_node_domain=True,
        ).to(self.device)
        calibrator.eval()
        backend = str(getattr(self, "recal_execution_backend", "reference"))
        engine = None
        if backend == "fused":
            if recal_feedback_mode != LEGACY_FEEDBACK or recal_maintenance_budget:
                raise ValueError(
                    "the fused backend supports legacy feedback without a Top-K budget"
                )
            engine = FusedReCal(
                calibrator,
                num_nodes=self.model.node_num,
                pred_len=self.output_len,
                mean=float(self.mean),
                std=max(float(self.std), 1e-08),
                batch_size=self.batch_size,
                device=self.device,
                lr=recal_lr,
                target_offset=recal_target_offset,
                update_interval=recal_update_interval,
                calendar=calendar,
                memory=memory_cfg,
            )
            assimilator = None
        else:
            optimizer = torch.optim.Adam(calibrator.parameters(), lr=recal_lr)
            assimilator = ForecastDiagonalAssimilator(
                calibrator=calibrator,
                optimizer=optimizer,
                mean=float(self.mean),
                std=max(float(self.std), 1e-08),
                device=self.device,
                target_offset=recal_target_offset,
                update_interval=recal_update_interval,
                maintenance_budget=recal_maintenance_budget,
                execution_backend=backend,
                feedback_mode=recal_feedback_mode,
                allowed_lateness=recal_allowed_lateness,
                calendar=calendar,
                memory=memory_cfg,
            )
        event_mode = recal_feedback_mode == EVENT_TIME_FEEDBACK
        feeder = None
        if event_mode:
            arrival = None
            if recal_feedback_trace:
                arrival = torch.as_tensor(
                    np.load(recal_feedback_trace)["arrival"], dtype=torch.long
                )
            feeder = StreamingTraceFeeder(
                self.model.node_num, arrival=arrival, target_offset=recal_target_offset
            )
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        infer_times = []
        wall = []
        total_samples = 0
        pred = []
        label = []
        num_val = self.testX.shape[0]
        num_batch = math.ceil(num_val / self.batch_size)
        for batch_idx in tqdm(range(num_batch)):
            if recal_max_batches and batch_idx >= recal_max_batches:
                break
            if not isinstance(self.model, torch.nn.Module):
                continue
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            wall_start = time.perf_counter()
            start_idx = batch_idx * self.batch_size
            end_idx = min(num_val, (batch_idx + 1) * self.batch_size)
            X = self.testX[start_idx:end_idx]
            Y = self.testY[start_idx:end_idx]
            TE = self._to_te_tensor(self.testXTE[start_idx:end_idx])
            NormX = self._to_norm_x_tensor(X)
            if engine is not None:
                infer_start = time.perf_counter()
                with torch.no_grad():
                    y_hat = self.model(NormX, TE).permute(0, 3, 2, 1)
                served = engine.step(start_idx, y_hat, self._to_label_tensor(Y))
                pred.append(served.unsqueeze(-1).cpu().numpy())
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                infer_times.append((time.perf_counter() - infer_start) * 1000)
                total_samples += end_idx - start_idx
                label.append(Y)
                wall.append(
                    ((time.perf_counter() - wall_start) * 1000, end_idx - start_idx)
                )
                continue
            if event_mode:
                for tau, nodes, vals, arr in feeder.pop_through(start_idx):
                    assimilator.observe(tau, nodes, vals, arrival_time=arr)
            assimilator.maintain(start_idx)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            infer_start = time.perf_counter()
            with torch.no_grad():
                y_hat = self.model(NormX, TE)
                y_hat = y_hat.permute(0, 3, 2, 1)
                calibrator.eval()
                calibrated = calibrator(y_hat)
                calibrated = calibrated.permute(0, 3, 2, 1)
                recalled = assimilator.memory_correction(start_idx, end_idx - start_idx)
                if recalled is not None:
                    served = calibrated * self.std + self.mean + recalled.unsqueeze(-1)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            infer_times.append((time.perf_counter() - infer_start) * 1000)
            total_samples += end_idx - start_idx
            if recalled is None:
                pred.append(calibrated.cpu().detach().numpy() * self.std + self.mean)
            else:
                pred.append(served.cpu().detach().numpy())
            label.append(Y)
            target = self._to_label_tensor(Y)
            if event_mode:
                feeder.push_labels(start_idx, target)
                assimilator.issue(start_idx, y_hat)
            else:
                assimilator.enqueue(start_idx, y_hat, target)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            wall.append(
                ((time.perf_counter() - wall_start) * 1000, end_idx - start_idx)
            )
        if event_mode:
            assimilator.finalize(num_val)
            es = assimilator.event_stats
            if es is not None:
                log_string(
                    log,
                    "Event Feedback Stats: arrived=%d consumed=%d duplicate=%d conflict=%d late=%d orphan=%d future_use=%d expired_node_slots=%d rounds=%d ws_groups_max=%d ws_node_cells_max=%d ws_bytes_max=%d age_mean=%.3f"
                    % (
                        es.arrived,
                        es.consumed,
                        es.duplicate,
                        es.conflict,
                        es.late,
                        es.orphan,
                        es.future_use,
                        es.expired_node_slots,
                        es.rounds,
                        es.max_pending_groups,
                        es.max_node_cells,
                        es.max_bytes,
                        es.age_mean,
                    ),
                )
        pred = np.concatenate(pred, axis=0)
        label = np.concatenate(label, axis=0)
        maes = []
        rmses = []
        mapes = []
        for i in range(pred.shape[1]):
            mae, rmse, mape = metric(pred[:, i, :], label[:, i, :])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(
                log,
                "step %d, mae: %.4f, rmse: %.4f, mape: %.4f" % (i + 1, mae, rmse, mape),
            )
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(
            log, "average, mae: %.4f, rmse: %.4f, mape: %.4f" % (mae, rmse, mape)
        )
        method_name = (
            f"RECAL(r={recal_rank})[offset={recal_target_offset}, interval={recal_update_interval}, budget={recal_maintenance_budget}"
            + (f", memory={memory_cfg.key}" if memory_cfg.enabled else "")
            + "]"
        )
        peak_mem = (
            torch.cuda.max_memory_allocated(self.device) / 1024**2
            if self.device.type == "cuda"
            else 0.0
        )
        total_infer_sec = sum(infer_times) / 1000.0
        samples_sec = total_samples / total_infer_sec if total_infer_sec > 0 else 0.0
        if engine is not None:
            self._log_efficiency_stats(
                method_name=method_name + "[fused]",
                n_params=count_trainable_params(calibrator),
                peak_mem_mb=peak_mem,
                samples_per_sec=samples_sec,
                adapt_ms=0.0,
                infer_ms=float(np.mean(infer_times)) if infer_times else 0.0,
                wall=wall,
            )
            log_string(log, "RECAL Fused Engine: %s" % engine.summary())
            return (np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0))
        self._log_efficiency_stats(
            method_name=method_name,
            n_params=count_trainable_params(calibrator),
            peak_mem_mb=peak_mem,
            samples_per_sec=samples_sec,
            adapt_ms=assimilator.stats.avg_adapt_ms,
            infer_ms=float(np.mean(infer_times)) if infer_times else 0.0,
            wall=wall,
        )
        log_string(
            log,
            "  Update Stats: applied=%d, eligible=%d, avg_cells=%.2f, avg_merged_diagonals=%.2f, avg_nodes_updated=%.2f, avg_eps_k=%.6f"
            % (
                assimilator.stats.applied,
                assimilator.stats.eligible,
                assimilator.stats.avg_cells,
                assimilator.stats.avg_merged_diagonals,
                assimilator.stats.avg_nodes_updated,
                assimilator.stats.avg_eps_k,
            ),
        )
        log_string(
            log,
            "  Lattice WorkingSet: max_cells=%d (N=%d, H=%d, N*H=%d)"
            % (
                getattr(assimilator, "max_working_set", 0),
                self.model.node_num,
                self.output_len,
                self.model.node_num * self.output_len,
            ),
        )
        log_string(log, "RECAL Execution Stats: %s" % assimilator.execution_stats)
        log_string(log, "RECAL Calendar Memory: %s" % assimilator.memory_stats)
        return (np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0))


def _normalize_cli_path(path: str) -> str:
    if path is None:
        return path
    if path == "":
        return path
    normalized = path.replace("\\", os.sep).replace("/", os.sep)
    return os.path.normpath(normalized)


def _infer_dataset_name(traffic_file: str) -> str:
    normalized = _normalize_cli_path(traffic_file)
    dataset_dir = os.path.basename(os.path.dirname(normalized))
    if not dataset_dir:
        raise ValueError(f"Cannot infer dataset name from traffic_file: {traffic_file}")
    return dataset_dir.lower()


def _canonical_artifact_paths(backbone: str, dataset: str):
    prefix = f"{backbone.lower()}_{dataset.lower()}"
    return (
        os.path.join(".", "cpt", f"{prefix}_model"),
        os.path.join(".", "log", f"{prefix}_log"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="configuration file")
    parser.add_argument("--backbone", type=str, default=None, choices=BACKBONES)
    args, _ = parser.parse_known_args()
    args.config = _normalize_cli_path(args.config)
    if not args.config:
        raise ValueError("`--config` is required, e.g. `--config ./config/SD.conf`.")
    config = configparser.ConfigParser()
    read_files = config.read(args.config)
    required_sections = ["train", "data", "param", "file"]
    if not read_files or any((section not in config for section in required_sections)):
        raise FileNotFoundError(
            f"Failed to load config from `{args.config}`. Please use a valid config path such as `./config/SD.conf` on Linux."
        )

    def _cfg(section, option, fallback):
        if config.has_section(section) and config.has_option(section, option):
            return config[section][option]
        return fallback

    config_backbone = _cfg("backbone", "name", "patchstg").lower()
    default_backbone = args.backbone or config_backbone
    if default_backbone not in BACKBONES:
        raise ValueError(
            f"Unsupported backbone: {default_backbone}. Use one of {BACKBONES}."
        )
    default_learning_rate = config["train"]["learning_rate"]
    default_weight_decay = config["train"]["weight_decay"]
    default_batch_size = config["train"]["batch_size"]
    default_max_epoch = config["train"]["max_epoch"]
    if default_backbone == "mage":
        default_learning_rate = _cfg("mage", "learning_rate", 0.01)
        default_weight_decay = _cfg("mage", "weight_decay", default_weight_decay)
        default_batch_size = _cfg("mage", "batch_size", 64)
        default_max_epoch = _cfg("mage", "max_epoch", default_max_epoch)
    parser.set_defaults(backbone=default_backbone)
    parser.add_argument("--cuda", type=str, default=config["train"]["cuda"])
    parser.add_argument("--seed", type=int, default=config["train"]["seed"])
    parser.add_argument("--batch_size", type=int, default=default_batch_size)
    parser.add_argument("--max_epoch", type=int, default=default_max_epoch)
    parser.add_argument("--learning_rate", type=float, default=default_learning_rate)
    parser.add_argument("--weight_decay", type=float, default=default_weight_decay)
    parser.add_argument("--input_len", type=int, default=config["data"]["input_len"])
    parser.add_argument("--output_len", type=int, default=config["data"]["output_len"])
    parser.add_argument(
        "--train_ratio", type=float, default=config["data"]["train_ratio"]
    )
    parser.add_argument("--val_ratio", type=float, default=config["data"]["val_ratio"])
    parser.add_argument(
        "--test_ratio", type=float, default=config["data"]["test_ratio"]
    )
    parser.add_argument("--layers", type=int, default=config["param"]["layers"])
    parser.add_argument("--tem_patchsize", type=int, default=config["param"]["tps"])
    parser.add_argument("--tem_patchnum", type=int, default=config["param"]["tpn"])
    parser.add_argument("--factors", type=int, default=config["param"]["factors"])
    parser.add_argument("--recur_times", type=int, default=config["param"]["recur"])
    parser.add_argument("--spa_patchsize", type=int, default=config["param"]["sps"])
    parser.add_argument("--spa_patchnum", type=int, default=config["param"]["spn"])
    parser.add_argument("--node_num", type=int, default=config["param"]["nodes"])
    parser.add_argument("--tod", type=int, default=config["param"]["tod"])
    parser.add_argument("--dow", type=int, default=config["param"]["dow"])
    parser.add_argument("--input_dims", type=int, default=config["param"]["id"])
    parser.add_argument("--node_dims", type=int, default=config["param"]["nd"])
    parser.add_argument("--tod_dims", type=int, default=config["param"]["td"])
    parser.add_argument("--dow_dims", type=int, default=config["param"]["dd"])
    parser.add_argument("--traffic_file", default=config["file"]["traffic"])
    parser.add_argument("--meta_file", default=config["file"]["meta"])
    parser.add_argument("--adj_file", default=config["file"]["adj"])
    parser.add_argument("--model_file", default=config["file"]["model"])
    parser.add_argument("--log_file", default=config["file"]["log"])
    parser.add_argument(
        "--legacy_model_file", default=_cfg("compat", "legacy_model", "")
    )
    parser.add_argument("--run_mode", type=str, default="test", choices=RUN_MODES)
    parser.add_argument("--test_mode", type=str, default="recal", choices=TEST_MODES)
    parser.add_argument(
        "--mage_model_dim", type=int, default=int(_cfg("mage", "model_dim", 128))
    )
    parser.add_argument(
        "--mage_recur_num", type=int, default=int(_cfg("mage", "recur_num", 16))
    )
    parser.add_argument("--mage_topk", type=int, default=int(_cfg("mage", "topk", 4)))
    parser.add_argument(
        "--mage_graph_gen_dim", type=int, default=int(_cfg("mage", "graph_gen_dim", 32))
    )
    parser.add_argument(
        "--mage_ffn_dropout",
        type=float,
        default=float(_cfg("mage", "ffn_dropout", 0.3)),
    )
    parser.add_argument(
        "--sqlinear_mlp_ratio",
        type=float,
        default=float(_cfg("sqlinear", "mlp_ratio", 3.0)),
    )
    parser.add_argument(
        "--sqlinear_node_rank", type=int, default=int(_cfg("sqlinear", "node_rank", 2))
    )
    parser.add_argument(
        "--feedback_delay", type=int, default=int(config["data"]["output_len"])
    )
    parser.add_argument("--calib_lr", type=float, default=0.0001)
    parser.add_argument("--recal_rank", type=int, default=4)
    parser.add_argument("--recal_lr", type=float, default=0.0001)
    parser.add_argument("--recal_target_offset", type=int, default=1)
    parser.add_argument("--recal_max_batches", type=int, default=0)
    parser.add_argument("--recal_update_interval", type=int, default=1)
    parser.add_argument("--recal_maintenance_budget", type=int, default=0)
    parser.add_argument(
        "--recal_execution_backend",
        default="reference",
        choices=("reference", "hardware_aware", "fused"),
        help="RECAL execution only (same method): hardware_aware = CUDA-graph C1+C2 update; fused = ring buffer + stacked memory + CUDA graphs for C1+C2+C3 (legacy feedback)",
    )
    parser.add_argument(
        "--recal_feedback_mode",
        type=str,
        default=LEGACY_FEEDBACK,
        choices=FEEDBACK_MODES,
    )
    parser.add_argument(
        "--recal_allowed_lateness",
        type=int,
        default=0,
        help="event_time only: maximum label wait L in ticks",
    )
    parser.add_argument("--recal_feedback_trace", type=str, default=None)
    parser.add_argument(
        "--recal_memory",
        type=str,
        default="auto_multi",
        choices=REQUEST_KEYS,
        help="calendar memory key; auto = decided on the train split (off = accepted RECAL)",
    )
    parser.add_argument("--recal_memory_decay", type=float, default=0.9)
    parser.add_argument("--recal_memory_shrink", type=float, default=4.0)
    parser.add_argument("--recal_memory_root_shrink", type=float, default=None)
    parser.add_argument("--recal_memory_joint", type=int, default=1, choices=(0, 1))
    parser.add_argument("--recal_memory_verify", type=int, default=1, choices=(0, 1))
    parser.add_argument("--recal_memory_verify_decay", type=float, default=0.998)
    parser.add_argument(
        "--recal_memory_verify_halflife_days",
        type=float,
        default=None,
        help="overrides verify_decay; 3.6 days gives 0.997996 per tick at 96 ticks/day (the SD anchor 15.0196 uses verify_decay 0.998 exactly)",
    )
    parser.add_argument(
        "--recal_memory_weekend",
        type=str,
        default="",
        help="comma-separated weekend dow indices; empty = inferred from the train split",
    )
    parser.add_argument(
        "--recal_memory_slot_ticks",
        type=int,
        default=1,
        help="ticks merged into one time-of-day slot",
    )
    parser.add_argument(
        "--recal_memory_null_tol",
        type=float,
        default=0.0,
        help="|label| <= tol counts as missing (LargeST zero-fill is exact 0)",
    )
    args = parser.parse_args()
    args.config = _normalize_cli_path(args.config)
    args.test_mode = normalize_test_mode(args.test_mode)
    for attr in [
        "traffic_file",
        "meta_file",
        "adj_file",
        "model_file",
        "log_file",
        "legacy_model_file",
    ]:
        setattr(args, attr, _normalize_cli_path(getattr(args, attr)))
    if args.backbone != "patchstg" and config_backbone == "patchstg":
        default_model_file = _normalize_cli_path(config["file"]["model"])
        default_log_file = _normalize_cli_path(config["file"]["log"])
        canonical_model_file, canonical_log_file = _canonical_artifact_paths(
            args.backbone, _infer_dataset_name(args.traffic_file)
        )
        if args.model_file == default_model_file:
            args.model_file = _normalize_cli_path(canonical_model_file)
        if args.log_file == default_log_file:
            args.log_file = _normalize_cli_path(canonical_log_file)
    log_dir = os.path.dirname(args.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    log = open(args.log_file, "w")
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
    log_string(log, "------------ Options -------------")
    for k, v in vars(args).items():
        log_string(log, "%s: %s" % (str(k), str(v)))
    log_string(log, "-------------- End ----------------")
    solver = Solver(vars(args))
    if args.run_mode in ("train", "both"):
        solver.train()
    if args.run_mode in ("test", "both"):
        solver.test_with_recal()
