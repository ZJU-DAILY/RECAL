import torch
import torch.nn as nn
import numpy as np
import sys
import os.path as osp
import networkx as nx
import torch.nn.functional as func
from tqdm import tqdm
from torch import optim
from datetime import datetime
import time
from torch_geometric.utils import to_dense_batch
from src.model.ewc import EWC
from torch_geometric.loader import DataLoader
from dataer.SpatioTemporalDataset import SpatioTemporalDataset
from utils.metric import cal_metric, masked_mae_np
from utils.common_tools import mkdirs, load_best_model

REPO_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[4])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from recal import (
    ReCalibrator,
    EVENT_TIME_FEEDBACK,
    ForecastDiagonalAssimilator,
    LEGACY_FEEDBACK,
    StreamingTraceFeeder,
    count_trainable_params,
)
from recal.fused_engine import FusedReCal
from trainer.calendar_source import continual_calendar_memory


def _prediction(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def test_model_with_recal(
    model,
    args,
    testset,
    pin_memory,
    recal_rank=4,
    recal_lr=0.0001,
    target_offset=1,
    max_batches=0,
    update_interval=1,
    maintenance_budget=0,
    feedback_mode=LEGACY_FEEDBACK,
    allowed_lateness=0,
    feedback_trace=None,
    execution_backend="reference",
):
    """RECAL: forecast-diagonal online adaptation for dense node forecasts."""
    model.eval()
    calibrator = None
    assimilator = None
    optimizer = None
    event_mode = feedback_mode == EVENT_TIME_FEEDBACK
    feeder = None
    calendar, memory = (None, None)
    source = getattr(args, "c3_source", None)
    if source is not None:
        calendar, memory = continual_calendar_memory(
            args,
            source["series"],
            source["train_span"],
            source["test_first_input_step"],
            source["test_y"],
            x_len=args.x_len,
            y_len=args.y_len,
            ticks_per_day=source["ticks_per_day"],
            logger=args.logger,
        )
    pred_ = []
    truth_ = []
    infer_times = []
    total_samples = 0
    origin_start = 0
    device_type = getattr(
        args.device, "type", "cuda" if str(args.device).startswith("cuda") else "cpu"
    )
    fused = execution_backend == "fused"
    if fused and (feedback_mode != LEGACY_FEEDBACK or maintenance_budget):
        raise ValueError(
            "the fused backend supports legacy feedback without a Top-K budget"
        )
    engine = None
    t_start = time.perf_counter()
    for batch_idx, data in enumerate(tqdm(testset, desc="RECAL TTA")):
        if max_batches and batch_idx >= max_batches:
            break
        data = data.to(args.device, non_blocking=pin_memory)
        if fused:
            if device_type == "cuda":
                torch.cuda.synchronize(args.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                raw_pred = _prediction(model(data, args.adj))
                raw_pred, _ = to_dense_batch(raw_pred, batch=data.batch)
                target_dense, _ = to_dense_batch(data.y, batch=data.batch)
                yb = raw_pred.unsqueeze(1)
            if engine is None:
                calibrator = ReCalibrator(
                    args.graph_size,
                    yb.shape[-1],
                    recal_rank,
                    use_horizon_domain=True,
                    use_node_domain=True,
                ).to(args.device)
                calibrator.eval()
                engine = FusedReCal(
                    calibrator,
                    num_nodes=args.graph_size,
                    pred_len=yb.shape[-1],
                    mean=0.0,
                    std=1.0,
                    batch_size=yb.shape[0],
                    device=args.device,
                    lr=recal_lr,
                    target_offset=target_offset,
                    update_interval=update_interval,
                    calendar=calendar,
                    memory=memory,
                )
            cal, recalled = engine.step(
                origin_start, yb, target_dense.permute(0, 2, 1), parts=True
            )
            yc = cal.squeeze(1)
            if recalled is not None:
                yc = yc + recalled.permute(0, 2, 1)
            if device_type == "cuda":
                torch.cuda.synchronize(args.device)
            infer_times.append((time.perf_counter() - t0) * 1000.0)
            pred_.append(yc.cpu().data.numpy())
            truth_.append(target_dense.cpu().data.numpy())
            total_samples += yc.shape[0]
            origin_start += yc.shape[0]
            continue
        if assimilator is not None:
            if event_mode:
                for tau, nodes, vals, arr in feeder.pop_through(origin_start):
                    assimilator.observe(tau, nodes, vals, arrival_time=arr)
            assimilator.maintain(origin_start)
        if device_type == "cuda":
            torch.cuda.synchronize(args.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            raw_pred = _prediction(model(data, args.adj))
            raw_pred, _ = to_dense_batch(raw_pred, batch=data.batch)
            target_dense, _ = to_dense_batch(data.y, batch=data.batch)
            yb = raw_pred.unsqueeze(1)
            if calibrator is None:
                pred_len = yb.shape[-1]
                calibrator = ReCalibrator(
                    args.graph_size,
                    pred_len,
                    recal_rank,
                    use_horizon_domain=True,
                    use_node_domain=True,
                ).to(args.device)
                calibrator.eval()
                optimizer = torch.optim.Adam(calibrator.parameters(), lr=recal_lr)
                assimilator = ForecastDiagonalAssimilator(
                    calibrator=calibrator,
                    optimizer=optimizer,
                    mean=0.0,
                    std=1.0,
                    device=args.device,
                    target_offset=target_offset,
                    update_interval=update_interval,
                    maintenance_budget=maintenance_budget,
                    execution_backend=execution_backend,
                    feedback_mode=feedback_mode,
                    allowed_lateness=int(allowed_lateness),
                    calendar=calendar,
                    memory=memory,
                )
                if event_mode:
                    arrival = None
                    if feedback_trace:
                        arrival = torch.as_tensor(
                            np.load(feedback_trace)["arrival"], dtype=torch.long
                        )
                    feeder = StreamingTraceFeeder(
                        args.graph_size, arrival=arrival, target_offset=target_offset
                    )
                args.logger.info(
                    "RECAL(r=%d), lr=%.2e, offset=%d, interval=%d, budget=%d",
                    recal_rank,
                    recal_lr,
                    target_offset,
                    update_interval,
                    maintenance_budget,
                )
            calibrator.eval()
            yc = calibrator(yb).squeeze(1)
            recalled = assimilator.memory_correction(origin_start, yc.shape[0])
            if recalled is not None:
                yc = yc + recalled.permute(0, 2, 1)
        if device_type == "cuda":
            torch.cuda.synchronize(args.device)
        infer_times.append((time.perf_counter() - t0) * 1000.0)
        pred_.append(yc.cpu().data.numpy())
        truth_.append(target_dense.cpu().data.numpy())
        total_samples += yc.shape[0]
        target = target_dense.permute(0, 2, 1).unsqueeze(-1)
        if event_mode:
            feeder.push_labels(origin_start, target)
            assimilator.issue(origin_start, yb)
        else:
            assimilator.enqueue(origin_start, yb, target)
        origin_start += yc.shape[0]
    if event_mode and assimilator is not None:
        assimilator.finalize(origin_start)
    t_end = time.perf_counter()
    pred_ = np.concatenate(pred_, 0)
    truth_ = np.concatenate(truth_, 0)
    cal_metric(truth_, pred_, args)
    infer_sec = sum(infer_times) / 1000.0
    samples_per_sec = total_samples / infer_sec if infer_sec > 0 else 0.0
    avg_infer_ms = float(np.mean(infer_times)) if infer_times else 0.0
    stats = assimilator.stats if assimilator is not None else None
    if assimilator is not None:
        args.logger.info("RECAL Execution Stats: %s", assimilator.execution_stats)
        args.logger.info("RECAL Calendar Memory: %s", assimilator.memory_stats)
    if engine is not None:
        args.logger.info("RECAL Fused Engine: %s", engine.summary())
    args.logger.info(
        "RECAL Efficiency: Params=%d, TotalTime=%.4fs, Samples/sec=%.2f, Infer ms/batch=%.4f, Adapt ms/update=%.4f",
        count_trainable_params(calibrator) if calibrator is not None else 0,
        t_end - t_start,
        samples_per_sec,
        avg_infer_ms,
        stats.avg_adapt_ms if stats is not None else 0.0,
    )
    args.logger.info(
        "RECAL Update Stats: applied=%d, eligible=%d, avg_cells=%.2f, avg_merged_diagonals=%.2f, avg_nodes_updated=%.2f, avg_eps_k=%.6f",
        stats.applied if stats is not None else 0,
        stats.eligible if stats is not None else 0,
        stats.avg_cells if stats is not None else 0.0,
        stats.avg_merged_diagonals if stats is not None else 0.0,
        stats.avg_nodes_updated if stats is not None else 0.0,
        stats.avg_eps_k if stats is not None else 0.0,
    )
