import os

import numpy as np
import pandas as pd
import torch


def resolve_data_root(explicit_root=None):
    if explicit_root:
        return os.path.abspath(explicit_root)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    return os.path.join(project_root, "data")


def resolve_dataset_root(dataset, data_root):
    dataset_key = str(dataset).strip().lower()
    candidates = [dataset_key, str(dataset).strip(), str(dataset).strip().upper()]
    for candidate in candidates:
        dataset_root = os.path.join(data_root, candidate)
        if os.path.isdir(dataset_root):
            return dataset_root, candidate.lower()
    raise FileNotFoundError(
        f"Dataset directory not found for {dataset!r} under {data_root}"
    )


def infer_node_num(dataset_root):
    year_dirs = sorted(
        entry
        for entry in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, entry)) and entry.isdigit()
    )
    if not year_dirs:
        raise FileNotFoundError(f"No yearly data folders found under {dataset_root}")

    sample_path = os.path.join(dataset_root, year_dirs[0], "his.npz")
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Missing history file: {sample_path}")

    ptr = np.load(sample_path)
    return int(ptr["data"].shape[1])


def get_dataset_info(dataset, data_root=None):
    data_root = resolve_data_root(data_root)
    dataset_root, dataset_key = resolve_dataset_root(dataset, data_root)
    node_num = infer_node_num(dataset_root)

    adj_path = os.path.join(dataset_root, f"{dataset_key}_rn_adj.npy")
    if not os.path.exists(adj_path):
        adj_candidates = sorted(
            entry for entry in os.listdir(dataset_root) if entry.endswith("_rn_adj.npy")
        )
        adj_path = (
            os.path.join(dataset_root, adj_candidates[0]) if adj_candidates else None
        )

    meta_path = os.path.join(dataset_root, f"{dataset_key}_meta.csv")
    if not os.path.exists(meta_path):
        meta_candidates = sorted(
            entry for entry in os.listdir(dataset_root) if entry.endswith("_meta.csv")
        )
        meta_path = (
            os.path.join(dataset_root, meta_candidates[0]) if meta_candidates else None
        )

    return dataset_root, adj_path, node_num, meta_path


def build_node_splits(node_num, args, logger):
    node_order = np.arange(node_num)
    split_seed = getattr(args, "split_seed", None)
    if split_seed is None:
        np.random.shuffle(node_order)
        split_seed_msg = "global"
    else:
        split_seed = int(split_seed)
        split_rng = np.random.RandomState(split_seed)
        split_rng.shuffle(node_order)
        split_seed_msg = str(split_seed)

    node_num_training = int(node_num / (1.0 + float(args.max_increase_ratio)))
    node_num_test_increase = int(node_num_training * float(args.test_increase_ratio))
    node_num_test_decrease = int(node_num_training * float(args.test_decrease_ratio))

    node_training = node_order[:node_num_training]
    node_test = np.concatenate(
        [
            node_order[: node_num_training - node_num_test_decrease],
            node_order[node_num_training : node_num_training + node_num_test_increase],
        ]
    )

    split = {
        "node_training": node_training.astype(np.int64, copy=False),
        "node_val": node_training.astype(np.int64, copy=False),
        "node_test": (
            node_test.astype(np.int64, copy=False)
            if bool(args.sood)
            else node_training.astype(np.int64, copy=False)
        ),
        "node_unseen_num": node_num_test_increase if bool(args.sood) else 0,
    }

    logger.info(
        "Node split summary: train_seen=%d, test_fix=%d, test_unseen=%d, split_seed=%s",
        len(split["node_training"]),
        len(split["node_test"]) - split["node_unseen_num"],
        split["node_unseen_num"],
        split_seed_msg,
    )
    return split


def haversine_distance_matrix(latitudes, longitudes):
    earth_radius_km = 6371.0
    lat_rad = np.radians(latitudes).reshape(-1, 1)
    lon_rad = np.radians(longitudes).reshape(-1, 1)
    dlat = lat_rad - lat_rad.T
    dlon = lon_rad - lon_rad.T
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_rad) * np.cos(lat_rad.T) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return earth_radius_km * c


def ensure_adjacency_matrix(adj_path, meta_path, node_num, logger=None, k=8):
    if adj_path and os.path.exists(adj_path):
        return adj_path
    if not meta_path or not os.path.exists(meta_path):
        return None

    meta = pd.read_csv(meta_path).iloc[:node_num].copy()
    if "Lat" not in meta.columns or "Lng" not in meta.columns:
        if logger is not None:
            logger.warning(
                "Missing Lat/Lng in metadata, cannot build adjacency: %s", meta_path
            )
        return None

    latitudes = pd.to_numeric(meta["Lat"], errors="coerce").to_numpy(dtype=np.float32)
    longitudes = pd.to_numeric(meta["Lng"], errors="coerce").to_numpy(dtype=np.float32)
    valid_mask = np.isfinite(latitudes) & np.isfinite(longitudes)
    if int(valid_mask.sum()) != node_num:
        if logger is not None:
            logger.warning(
                "Invalid Lat/Lng rows in metadata, cannot build adjacency: %s",
                meta_path,
            )
        return None

    distance_mx = haversine_distance_matrix(latitudes, longitudes)
    np.fill_diagonal(distance_mx, np.inf)
    neighbor_count = max(1, min(int(k), node_num - 1))
    neighbor_idx = np.argpartition(distance_mx, kth=neighbor_count - 1, axis=1)[
        :, :neighbor_count
    ]

    adjacency = np.zeros((node_num, node_num), dtype=np.float32)
    row_idx = np.arange(node_num)[:, None]
    adjacency[row_idx, neighbor_idx] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)

    if adj_path:
        np.save(adj_path, adjacency)
        if logger is not None:
            logger.info(
                "Built adjacency from metadata and saved to %s with k=%d",
                adj_path,
                neighbor_count,
            )
        return adj_path
    return None


def load_node_metadata(meta_path, node_num, logger=None, fit_indices=None):
    if not meta_path or not os.path.exists(meta_path):
        if logger is not None:
            logger.info("Node metadata not found, skip metadata prior.")
        return None

    meta = pd.read_csv(meta_path).iloc[:node_num].copy()
    if len(meta) != node_num:
        if logger is not None:
            logger.warning(
                "Metadata row count %d does not match node count %d, skip metadata prior.",
                len(meta),
                node_num,
            )
        return None

    if fit_indices is None:
        fit_indices = np.arange(node_num)
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    fit_meta = meta.iloc[fit_indices].copy()

    feature_blocks = []

    numeric = meta.select_dtypes(include=[np.number]).copy()
    fit_numeric = fit_meta.select_dtypes(include=[np.number]).copy()
    drop_numeric = [
        col
        for col in numeric.columns
        if col.lower().startswith("id") or numeric[col].nunique(dropna=False) <= 1
    ]
    numeric = numeric.drop(columns=drop_numeric, errors="ignore")
    fit_numeric = fit_numeric.drop(columns=drop_numeric, errors="ignore")

    if not numeric.empty:
        fit_numeric = fit_numeric.fillna(fit_numeric.median())
        numeric = numeric.fillna(fit_numeric.median())
        numeric_array = numeric.to_numpy(dtype=np.float32)
        fit_numeric_array = fit_numeric.to_numpy(dtype=np.float32)
        numeric_mean = fit_numeric_array.mean(axis=0, keepdims=True)
        numeric_std = fit_numeric_array.std(axis=0, keepdims=True)
        numeric_std[numeric_std < 1e-6] = 1.0
        feature_blocks.append((numeric_array - numeric_mean) / numeric_std)

    categorical_cols = []
    for col in meta.columns:
        if col in numeric.columns or col.lower().startswith("id"):
            continue
        if meta[col].dtype == object or str(meta[col].dtype).startswith("category"):
            categorical_cols.append(col)

    if categorical_cols:
        fit_categorical = fit_meta[categorical_cols].fillna("UNK").copy()
        categorical = meta[categorical_cols].fillna("UNK").copy()
        for col in categorical_cols:
            seen_categories = set(fit_categorical[col].astype(str).tolist())
            categorical[col] = (
                categorical[col]
                .astype(str)
                .where(categorical[col].astype(str).isin(seen_categories), "UNK")
            )
        fit_columns = pd.get_dummies(
            fit_categorical, dummy_na=False, dtype=np.float32
        ).columns
        categorical = pd.get_dummies(categorical, dummy_na=False, dtype=np.float32)
        categorical = categorical.reindex(columns=fit_columns, fill_value=0.0)
        feature_blocks.append(categorical.to_numpy(dtype=np.float32))

    if not feature_blocks:
        if logger is not None:
            logger.info("Metadata file has no usable columns, skip metadata prior.")
        return None

    features = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    if logger is not None:
        logger.info("Loaded node metadata: %s -> %s", meta_path, features.shape)
    return torch.tensor(features, dtype=torch.float32)


def build_static_prior(
    node_num,
    adj_path,
    meta_path,
    use_meta,
    use_degree,
    logger,
    fit_indices=None,
):
    if fit_indices is None:
        fit_indices = np.arange(node_num)
    fit_indices = np.asarray(fit_indices, dtype=np.int64)

    blocks = []

    if bool(use_meta):
        node_meta = load_node_metadata(
            meta_path=meta_path,
            node_num=node_num,
            logger=logger,
            fit_indices=fit_indices,
        )
        if node_meta is not None:
            blocks.append(node_meta.float())

    resolved_adj_path = ensure_adjacency_matrix(
        adj_path=adj_path,
        meta_path=meta_path,
        node_num=node_num,
        logger=logger,
    )
    if bool(use_degree) and resolved_adj_path and os.path.exists(resolved_adj_path):
        adj = np.load(resolved_adj_path)
        degree = np.asarray(adj, dtype=np.float32).sum(axis=1, keepdims=True)
        degree = np.log1p(degree)
        fit_degree = degree[fit_indices]
        degree_max = float(fit_degree.max()) if fit_degree.size else 0.0
        if degree_max > 0:
            degree = degree / degree_max
        blocks.append(torch.tensor(degree, dtype=torch.float32))

    if not blocks:
        logger.info("Static prior disabled: no metadata/degree features loaded.")
        return None

    static_prior = torch.cat(blocks, dim=-1)
    logger.info("Static prior shape: %s", tuple(static_prior.shape))
    return static_prior
