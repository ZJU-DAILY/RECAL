import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from data import (
    build_node_splits,
    build_static_prior,
    ensure_adjacency_matrix,
    get_dataset_info,
)

from metrics import compute_all_metrics
from models.react import React


def set_seed(seed):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(args, node_num, static_prior, adj_mx):
    return React(
        node_num=node_num,
        input_dim=int(args.input_dim),
        output_dim=int(args.output_dim),
        seq_len=int(args.seq_len),
        horizon=int(args.horizon),
        embed_dim=int(args.embed_dim),
        temp_dim_tid=int(args.temp_dim_tid),
        temp_dim_diw=int(args.temp_dim_diw),
        num_layer=int(args.num_layer),
        mlp_dropout=float(args.dropout),
        steps_per_day=int(args.steps_per_day),
        days_per_week=int(args.days_per_week),
        use_tod_embedding=bool(args.use_tod_embedding),
        use_dow_embedding=bool(args.use_dow_embedding),
        static_prior=static_prior,
        adj_mx=adj_mx,
        context_dim=int(args.context_dim),
        context_hidden=int(args.context_hidden),
        support_mode=str(getattr(args, "support_mode", "entropy")),
        support_mix=float(getattr(args, "support_mix", 0.1)),
        support_mix_min=float(getattr(args, "support_mix_min", 0.05)),
        context_reg_weight=float(getattr(args, "context_reg_weight", 0.0)),
        use_context=bool(getattr(args, "use_context", 1)),
        use_dynamic_context=bool(getattr(args, "use_dynamic_context", 1)),
        use_context_gate=bool(getattr(args, "use_context_gate", 1)),
    )


def override_eval_args(base_args, dataset, device, scenario, data_root=None):
    values = vars(base_args).copy()
    values.update(
        {
            "data_root": (
                data_root
                if data_root is not None
                else getattr(base_args, "data_root", None)
            ),
            "dataset": dataset,
            "device": device,
            "years": "2017",
            "checkyears": scenario["checkyears"],
            "mode": "test",
            "seed": int(base_args.seed),
            "max_increase_ratio": 0.3,
            "test_increase_ratio": float(scenario["inc"]),
            "test_decrease_ratio": float(scenario["dec"]),
            "sood": int(scenario["sood"]),
            "tood": int(scenario["tood"]),
            "checkall": 1,
        }
    )
    return SimpleNamespace(**values)


def mask_value(labels):
    value = torch.tensor(0.0)
    if labels.min() < 1:
        value = labels.min()
    return value


def metric_block(preds, labels, horizon):
    mv = mask_value(labels)
    maes = []
    mapes = []
    rmses = []
    per_horizon = {}
    for idx in range(horizon):
        mae, mape, rmse = compute_all_metrics(preds[:, idx, :], labels[:, idx, :], mv)
        maes.append(float(mae))
        mapes.append(float(mape))
        rmses.append(float(rmse))
        if idx in {2, 5, 11}:
            per_horizon["h{}".format(idx + 1)] = {
                "mae": float(mae),
                "rmse": float(rmse),
                "mape": float(mape),
            }
    return {
        "avg": {
            "mae": float(np.mean(maes)),
            "rmse": float(np.mean(rmses)),
            "mape": float(np.mean(mapes)),
        },
        **per_horizon,
    }


def compute_metrics(preds, labels, node_sood_num, horizon):
    fix_preds = preds[..., :-node_sood_num] if node_sood_num else preds
    fix_labels = labels[..., :-node_sood_num] if node_sood_num else labels
    metrics = {
        "all": metric_block(preds, labels, horizon),
        "fix": metric_block(fix_preds, fix_labels, horizon),
    }
    if node_sood_num:
        unf_preds = preds[..., -node_sood_num:]
        unf_labels = labels[..., -node_sood_num:]
        metrics["unf"] = metric_block(unf_preds, unf_labels, horizon)
    return metrics


def load_graph_resources(dataset, args, logger, cache, node_num, train_nodes):
    key = (
        str(dataset),
        str(getattr(args, "data_root", "")),
        int(args.seed),
        None if getattr(args, "split_seed", None) is None else int(args.split_seed),
        float(args.max_increase_ratio),
        int(bool(args.use_meta_prior)),
        int(bool(args.use_degree_prior)),
        str(args.support_mode),
    )
    if key not in cache:
        _, adj_path, _, meta_path = get_dataset_info(
            dataset, getattr(args, "data_root", None)
        )
        static_prior = build_static_prior(
            node_num=node_num,
            adj_path=adj_path,
            meta_path=meta_path,
            use_meta=bool(args.use_meta_prior),
            use_degree=bool(args.use_degree_prior),
            logger=logger,
            fit_indices=train_nodes,
        )
        resolved_adj_path = ensure_adjacency_matrix(
            adj_path=adj_path, meta_path=meta_path, node_num=node_num, logger=logger
        )
        adj_mx = None
        if (
            str(args.support_mode) != "none"
            and resolved_adj_path
            and os.path.exists(resolved_adj_path)
        ):
            adj_mx = torch.tensor(np.load(resolved_adj_path), dtype=torch.float32)
            logger.info("Loaded adjacency for context support: %s", tuple(adj_mx.shape))
        elif str(args.support_mode) != "none":
            logger.warning(
                "Support mode %s requested but adjacency is unavailable.",
                args.support_mode,
            )
        cache[key] = {"static_prior": static_prior, "adj_mx": adj_mx}
    return cache[key]


def model_args(dataset):
    """Construct the architecture associated with the released checkpoint."""
    values = {
        "bs": 64,
        "checkall": 0,
        "checkyears": "2018",
        "context_reg_weight": 0.0,
        "days_per_week": 7,
        "dropout": 0.1,
        "embed_dim": 48,
        "horizon": 12,
        "input_dim": 3,
        "max_increase_ratio": 0.3,
        "mode": "train",
        "num_layer": 1,
        "output_dim": 1,
        "seed": 3407,
        "seq_len": 12,
        "sood": 1,
        "split_seed": None,
        "steps_per_day": 96,
        "support_mode": "entropy",
        "test_decrease_ratio": 0.1,
        "test_increase_ratio": 0.1,
        "tood": 1,
        "use_context": 1,
        "use_context_gate": 1,
        "use_degree_prior": 1,
        "use_dow_embedding": 1,
        "use_dynamic_context": 1,
        "use_meta_prior": 1,
        "use_tod_embedding": 1,
        "years": "2017",
    }
    values.update(
        {
            "SD": {
                "dataset": "SD",
                "context_dim": 32,
                "context_hidden": 64,
                "node_mask_ratio": 0.1,
                "support_mix": 0.2,
                "support_mix_min": 0.05,
                "temp_dim_diw": 48,
                "temp_dim_tid": 48,
            },
            "GBA": {
                "dataset": "GBA",
                "context_dim": 48,
                "context_hidden": 96,
                "node_mask_ratio": 0.15,
                "support_mix": 0.15,
                "support_mix_min": 0.0,
                "temp_dim_diw": 32,
                "temp_dim_tid": 32,
            },
        }[dataset]
    )
    return SimpleNamespace(**values)
