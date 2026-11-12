import heapq
import math

import numpy as np
import torch


def _build_graph(adj_mx):
    rows, cols = np.nonzero(adj_mx > 0)
    graph = [[] for _ in range(adj_mx.shape[0])]
    for src, dst in zip(rows.tolist(), cols.tolist()):
        if src == dst:
            continue
        weight = float(adj_mx[src, dst])
        if not np.isfinite(weight) or weight <= 0:
            continue
        graph[src].append((dst, weight))
    return graph


def _dijkstra(graph, start):
    node_num = len(graph)
    dist = np.full(node_num, np.inf, dtype=np.float32)
    dist[start] = 0.0
    heap = [(0.0, start)]

    while heap:
        cur_dist, node = heapq.heappop(heap)
        if cur_dist > dist[node]:
            continue
        for nxt, weight in graph[node]:
            new_dist = cur_dist + weight
            if new_dist < dist[nxt]:
                dist[nxt] = new_dist
                heapq.heappush(heap, (new_dist, nxt))
    return dist


def compute_spatial_embedding(
    adj_mx, node_num_ob, c=1, seed=0, device="cpu", target_dim=None
):
    adj_mx = np.asarray(adj_mx, dtype=np.float32).copy()
    np.fill_diagonal(adj_mx, 0.0)

    node_num = adj_mx.shape[0]
    node_num_ob = int(max(1, min(node_num_ob, node_num)))

    if target_dim is None:
        sem_dim = max(1, int(math.ceil(math.log2(max(2, node_num_ob)))))
    else:
        sem_dim = int(target_dim)

    landmark_pool = np.arange(node_num_ob, dtype=np.int64)
    if landmark_pool.size == 0:
        landmark_pool = np.arange(node_num, dtype=np.int64)

    rng = np.random.RandomState(int(seed) + int(c))
    replace = landmark_pool.size < sem_dim
    landmarks = rng.choice(landmark_pool, size=sem_dim, replace=replace)

    graph = _build_graph(adj_mx)
    features = np.zeros((node_num, sem_dim), dtype=np.float32)
    default_fill = float(adj_mx[adj_mx > 0].max()) if np.any(adj_mx > 0) else 1.0

    for idx, landmark in enumerate(landmarks.tolist()):
        dist = _dijkstra(graph, int(landmark))
        finite_mask = np.isfinite(dist)
        if finite_mask.any():
            max_dist = float(dist[finite_mask].max())
            fill_value = max_dist if max_dist > 0 else default_fill
            dist[~finite_mask] = fill_value
        else:
            dist.fill(default_fill)
        features[:, idx] = np.log1p(dist)

    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    features = (features - mean) / (std + 1e-6)
    return torch.tensor(features, dtype=torch.float32, device=device)
