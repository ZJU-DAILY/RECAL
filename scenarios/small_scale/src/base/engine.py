import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm

REPO_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[4])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from recal import (
    ReCalibrator,
    EVENT_TIME_FEEDBACK,
    ForecastDiagonalAssimilator,
    LEGACY_FEEDBACK,
    StreamingTraceFeeder,
    calendar_from_series,
    count_trainable_params,
    memory_config_from_options,
)
from recal.fused_engine import FusedReCal
from src.utils.metrics import masked_mape
from src.utils.metrics import masked_rmse
from src.utils.metrics import compute_all_metrics
import torch.nn as nn


class BaseEngine:

    def __init__(
        self,
        device,
        model,
        dataloader,
        scaler,
        sampler,
        loss_fn,
        lrate,
        optimizer,
        scheduler,
        clip_grad_value,
        max_epochs,
        patience,
        log_dir,
        logger,
        eval_method,
        seed,
    ):
        super().__init__()
        self._device = device
        self.model = model
        self.model.to(self._device)
        self._dataloader = dataloader
        self._scaler = scaler
        self._loss_fn = loss_fn
        self._lrate = lrate
        self._optimizer = optimizer
        self._lr_scheduler = scheduler
        self._clip_grad_value = clip_grad_value
        self._max_epochs = max_epochs
        self._patience = patience
        self._iter_cnt = 0
        self._save_path = log_dir
        self._logger = logger
        self._eval_method = eval_method
        self._seed = seed
        self._logger.info("The arc of model:\n{}".format(self.model))
        num_param = sum((p.numel() for p in model.parameters() if p.requires_grad))
        num_total_param = sum((p.numel() for p in model.parameters()))
        self._logger.info(
            "Number of total parameters: {}, tunable parameters: {}".format(
                num_total_param, num_param
            )
        )
        self._logger.info("The number of parameters: {}".format(self.model.param_num()))

    def _to_device(self, tensors):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)

    def _to_numpy(self, tensors):
        if isinstance(tensors, list):
            return [tensor.detach().cpu().numpy() for tensor in tensors]
        else:
            return tensors.detach().cpu().numpy()

    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [torch.tensor(array, dtype=torch.float32) for array in nparray]
        else:
            return torch.tensor(nparray, dtype=torch.float32)

    def _inverse_transform(self, tensors):

        def inv(tensor):
            return self._scaler.inverse_transform(tensor)

        if isinstance(tensors, list):
            return [inv(tensor) for tensor in tensors]
        else:
            return inv(tensors)

    def _scaler_float(self, name):
        value = getattr(self._scaler, name)
        if torch.is_tensor(value):
            value = value.detach().cpu()
            return float(value.reshape(-1)[0])
        array = np.asarray(value)
        return float(array.reshape(-1)[0])

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = "final_model_s{}.pt".format(self._seed)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))

    def load_model(self, save_path):
        filename = "final_model_s{}.pt".format(self._seed)
        if os.environ.get("RECAL_SOURCE_CHECKPOINT"):
            from scenarios.zero_shot.transfer import transfer_state

            state, fraction = transfer_state(
                torch.load(os.environ["RECAL_SOURCE_CHECKPOINT"], map_location="cpu"),
                self.model.state_dict(),
            )
            self._logger.info("Transferred parameter fraction: %.6f", fraction)
            self.model.load_state_dict(state)
        else:
            self.model.load_state_dict(
                torch.load(
                    os.environ.get(
                        "RECAL_CHECKPOINT", os.path.join(save_path, filename)
                    ),
                    map_location=self._device,
                )
            )

    def train_batch(self):
        self.model.train()
        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader["train_loader"].shuffle()
        for X, label in tqdm(self._dataloader["train_loader"].get_iterator()):
            self._optimizer.zero_grad()
            X, label = self._to_device(self._to_tensor([X, label]))
            pred = self.model(X, label)
            pred, label = self._inverse_transform([pred, label])
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print("Check mask value", mask_value)
            loss = self._loss_fn(pred, label, mask_value)
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()
            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._clip_grad_value
                )
            self._optimizer.step()
            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)
            self._iter_cnt += 1
        return (np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse))

    def train(self):
        self._logger.info("Start training!")
        wait = 0
        min_loss = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse = self.train_batch()
            t2 = time.time()
            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = self.evaluate_backbone("val")
            v2 = time.time()
            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()
            message = "Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}"
            self._logger.info(
                message.format(
                    epoch + 1,
                    mtrain_loss,
                    mtrain_rmse,
                    mtrain_mape,
                    mvalid_loss,
                    mvalid_rmse,
                    mvalid_mape,
                    t2 - t1,
                    v2 - v1,
                    cur_lr,
                )
            )
            if mvalid_loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info(
                    "Val loss decrease from {:.4f} to {:.4f}".format(
                        min_loss, mvalid_loss
                    )
                )
                min_loss = mvalid_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info(
                        "Early stop at epoch {}, loss = {:.6f}".format(
                            epoch + 1, min_loss
                        )
                    )
                    break
        self.evaluate_backbone("test")

    def evaluate_with_recal(
        self,
        mode,
        recal_rank=4,
        recal_lr=0.0001,
        recal_target_offset=1,
        recal_max_batches=0,
        recal_update_interval=1,
        recal_maintenance_budget=0,
        recal_feedback_mode=LEGACY_FEEDBACK,
        recal_allowed_lateness=0,
        recal_feedback_trace=None,
        recal_execution_backend="reference",
        recal_memory="off",
        recal_memory_source=None,
        recal_memory_options=None,
        recal_memory_slot_ticks=1,
        recal_memory_weekend=None,
    ):
        if mode == "test":
            self.load_model(self._save_path)
        self.model.eval()
        if mode == "val":
            preds = []
            labels = []
            with torch.no_grad():
                for X, label in tqdm(self._dataloader[mode + "_loader"].get_iterator()):
                    X, label = self._to_device(self._to_tensor([X, label]))
                    pred = self.model(X, label)
                    pred, label = self._inverse_transform([pred, label])
                    preds.append(pred.squeeze(-1).cpu())
                    labels.append(label.squeeze(-1).cpu())
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)
            mask_value = torch.tensor(0)
            if labels.min() < 1:
                mask_value = labels.min()
        else:
            pred_len = self.model.horizon
            calibrator = ReCalibrator(
                self.model.node_num,
                pred_len,
                recal_rank,
                use_horizon_domain=True,
                use_node_domain=True,
            ).to(self._device)
            calibrator.eval()
            optim_cal = torch.optim.Adam(calibrator.parameters(), lr=recal_lr)
            mean = self._scaler_float("mean")
            std = max(self._scaler_float("std"), 1e-08)
            calendar, memory_cfg = (None, None)
            if recal_memory != "off":
                src = recal_memory_source
                calendar, memory_key, weekend, score = calendar_from_series(
                    src["values"],
                    src["tod"],
                    src["dow"],
                    src["ticks_per_day"],
                    recal_memory,
                    src["select_span"],
                    src["first_input_step"],
                    src["num_samples"],
                    seq_len=self._dataloader[mode + "_loader"].seq_len,
                    horizon=pred_len,
                    slot_ticks=recal_memory_slot_ticks,
                    target_offset=recal_target_offset,
                    weekend_override=recal_memory_weekend,
                )
                memory_cfg = memory_config_from_options(
                    memory_key, src["ticks_per_day"], **recal_memory_options or {}
                )
                self._logger.info(
                    "RECAL calendar: key=%s (requested %s), weekend_dows=%s, weekly_score=%.2f, ticks_per_day=%d, slot_ticks=%d",
                    memory_key,
                    recal_memory,
                    weekend,
                    score,
                    src["ticks_per_day"],
                    recal_memory_slot_ticks,
                )
            fused = recal_execution_backend == "fused"
            if fused and (
                recal_feedback_mode != LEGACY_FEEDBACK or int(recal_maintenance_budget)
            ):
                raise ValueError(
                    "the fused backend supports legacy feedback without a Top-K budget"
                )
            engine, assimilator = (None, None)
            if not fused:
                assimilator = ForecastDiagonalAssimilator(
                    calibrator=calibrator,
                    optimizer=optim_cal,
                    mean=mean,
                    std=std,
                    device=self._device,
                    target_offset=recal_target_offset,
                    update_interval=int(recal_update_interval),
                    maintenance_budget=int(recal_maintenance_budget),
                    execution_backend=recal_execution_backend,
                    feedback_mode=recal_feedback_mode,
                    allowed_lateness=int(recal_allowed_lateness),
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
                    self.model.node_num,
                    arrival=arrival,
                    target_offset=recal_target_offset,
                )
            preds, labels = ([], [])
            infer_times = []
            total_samples = 0
            origin_start = 0
            device_type = getattr(
                self._device,
                "type",
                "cuda" if str(self._device).startswith("cuda") else "cpu",
            )
            t1 = time.perf_counter()
            for batch_idx, (x, y) in enumerate(
                tqdm(
                    self._dataloader[mode + "_loader"].get_iterator(), desc="RECAL TTA"
                )
            ):
                if recal_max_batches and batch_idx >= recal_max_batches:
                    break
                x, y = self._to_device(self._to_tensor([x, y]))
                if fused:
                    if engine is None:
                        engine = FusedReCal(
                            calibrator,
                            num_nodes=self.model.node_num,
                            pred_len=pred_len,
                            mean=mean,
                            std=std,
                            batch_size=x.shape[0],
                            device=self._device,
                            lr=recal_lr,
                            target_offset=recal_target_offset,
                            update_interval=int(recal_update_interval),
                            calendar=calendar,
                            memory=memory_cfg,
                        )
                    if device_type == "cuda":
                        torch.cuda.synchronize(self._device)
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        raw_pred = self.model(x).permute(0, 3, 2, 1)
                    [label] = self._inverse_transform([y])
                    cal, recalled = engine.step(
                        origin_start, raw_pred, label, parts=True
                    )
                    cal_pred = cal.permute(0, 3, 2, 1)
                    if device_type == "cuda":
                        torch.cuda.synchronize(self._device)
                    infer_times.append((time.perf_counter() - t0) * 1000.0)
                    [pred] = self._inverse_transform([cal_pred])
                    if recalled is not None:
                        pred = pred + recalled.unsqueeze(-1).to(pred.device)
                    preds.append(pred.squeeze(-1).cpu())
                    labels.append(label.squeeze(-1).cpu())
                    total_samples += x.shape[0]
                    origin_start += x.shape[0]
                    continue
                if event_mode:
                    [label_orig] = self._inverse_transform([y])
                    feeder.push_labels(origin_start, label_orig)
                    for tau, nodes, vals, arr in feeder.pop_through(origin_start):
                        assimilator.observe(tau, nodes, vals, arrival_time=arr)
                assimilator.maintain(origin_start)
                if device_type == "cuda":
                    torch.cuda.synchronize(self._device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    raw_pred = self.model(x)
                    raw_pred = raw_pred.permute(0, 3, 2, 1)
                    calibrator.eval()
                    cal_pred = calibrator(raw_pred)
                    cal_pred = cal_pred.permute(0, 3, 2, 1)
                    recalled = assimilator.memory_correction(origin_start, x.shape[0])
                if device_type == "cuda":
                    torch.cuda.synchronize(self._device)
                infer_times.append((time.perf_counter() - t0) * 1000.0)
                pred, label = self._inverse_transform([cal_pred, y])
                if recalled is not None:
                    pred = pred + recalled.unsqueeze(-1).to(pred.device)
                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())
                if event_mode:
                    assimilator.issue(origin_start, raw_pred)
                else:
                    assimilator.enqueue(origin_start, raw_pred, label)
                batch_size = x.shape[0]
                total_samples += batch_size
                origin_start += batch_size
            t2 = time.perf_counter()
            if event_mode:
                assimilator.finalize(origin_start)
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)
            mask_value = torch.tensor(0)
            if labels.min() < 1:
                mask_value = labels.min()
        if mode == "val":
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return (mae, mape, rmse)
        elif mode == "test":
            test_mae = []
            test_mape = []
            test_rmse = []
            print("Check mask value", mask_value)
            method_name = (
                f"RECAL(r={recal_rank})[offset={recal_target_offset}, interval={recal_update_interval}, budget={recal_maintenance_budget}"
                + (f", memory={memory_cfg.key}" if memory_cfg is not None else "")
                + "]"
            )
            self._logger.info("-" * 50 + f"{method_name} Test Time Results:" + "-" * 50)
            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:, i, :], labels[:, i, :], mask_value)
                log = "Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}"
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])
            avg_infer_ms = float(np.mean(infer_times)) if infer_times else 0.0
            infer_sec = sum(infer_times) / 1000.0
            samples_per_sec = total_samples / infer_sec if infer_sec > 0 else 0.0
            log = "Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}"
            self._logger.info(
                log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape))
            )
            self._logger.info(
                "%s Test Time: %.4fs, Params: %d",
                method_name,
                t2 - t1,
                count_trainable_params(calibrator),
            )
            self._logger.info(
                "%s Efficiency: TotalTime=%.4fs, Samples/sec=%.2f, Infer ms/batch=%.4f, Adapt ms/update=%.4f",
                method_name,
                t2 - t1,
                samples_per_sec,
                avg_infer_ms,
                assimilator.stats.avg_adapt_ms if assimilator is not None else 0.0,
            )
            if device_type == "cuda":
                self._logger.info(
                    "%s PeakMem MB=%.2f",
                    method_name,
                    torch.cuda.max_memory_allocated(self._device) / (1024.0 * 1024.0),
                )
            if assimilator is None:
                self._logger.info(
                    "RECAL Fused Engine: %s",
                    engine.summary() if engine is not None else {},
                )
                return
            self._logger.info(
                "%s Update Stats: applied=%d, eligible=%d, avg_cells=%.2f, avg_merged_diagonals=%.2f, avg_nodes_updated=%.2f, avg_eps_k=%.6f",
                method_name,
                assimilator.stats.applied,
                assimilator.stats.eligible,
                assimilator.stats.avg_cells,
                assimilator.stats.avg_merged_diagonals,
                assimilator.stats.avg_nodes_updated,
                assimilator.stats.avg_eps_k,
            )
            self._logger.info(
                "%s Lattice WorkingSet: max_cells=%d (N=%d, H=%d, N*H=%d)",
                method_name,
                getattr(assimilator, "max_working_set", 0),
                self.model.node_num,
                self.model.horizon,
                self.model.node_num * self.model.horizon,
            )
            self._logger.info("RECAL Execution Stats: %s", assimilator.execution_stats)
            self._logger.info("RECAL Calendar Memory: %s", assimilator.memory_stats)

    def evaluate_backbone(self, mode):
        if mode == "test":
            self.load_model(self._save_path)
        self.model.eval()
        preds = []
        labels = []
        infer_times = []
        total_samples = 0
        device_type = getattr(
            self._device,
            "type",
            "cuda" if str(self._device).startswith("cuda") else "cpu",
        )
        t1 = time.perf_counter()
        with torch.no_grad():
            for X, label in tqdm(self._dataloader[mode + "_loader"].get_iterator()):
                X, label = self._to_device(self._to_tensor([X, label]))
                if device_type == "cuda":
                    torch.cuda.synchronize(self._device)
                t0 = time.perf_counter()
                pred = self.model(X, label)
                pred, label = self._inverse_transform([pred, label])
                if device_type == "cuda":
                    torch.cuda.synchronize(self._device)
                infer_times.append((time.perf_counter() - t0) * 1000.0)
                total_samples += X.shape[0]
                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())
        t2 = time.perf_counter()
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()
        if mode == "val":
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return (mae, mape, rmse)
        elif mode == "test":
            test_mae = []
            test_mape = []
            test_rmse = []
            print("Check mask value", mask_value)
            self._logger.info("-" * 50 + "Base Test Time Results:" + "-" * 50)
            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:, i, :], labels[:, i, :], mask_value)
                log = "Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}"
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])
            log = "Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}"
            self._logger.info(
                log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape))
            )
            self._logger.info("Base Test Time: {:.4f}s".format(t2 - t1))
            infer_sec = sum(infer_times) / 1000.0
            samples_per_sec = total_samples / infer_sec if infer_sec > 0 else 0.0
            avg_infer_ms = float(np.mean(infer_times)) if infer_times else 0.0
            self._logger.info(
                "Baseline Efficiency: Params=0, TotalTime=%.4fs, Samples/sec=%.2f, Infer ms/batch=%.4f, Adapt ms/update=0.0000",
                t2 - t1,
                samples_per_sec,
                avg_infer_ms,
            )
            if device_type == "cuda":
                self._logger.info(
                    "Baseline PeakMem MB=%.2f",
                    torch.cuda.max_memory_allocated(self._device) / (1024.0 * 1024.0),
                )
