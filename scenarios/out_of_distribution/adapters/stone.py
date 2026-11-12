import math
import numpy as np
import torch
from types import SimpleNamespace
from models.stone import STONE
from models.spatial import compute_spatial_embedding


def build_split(node_num, args):
    np.random.seed(args.seed)
    node_order = np.arange(node_num)
    np.random.shuffle(node_order)
    node_num_training = int(node_num / (1.0 + args.max_increase_ratio))
    node_num_test_increase = int(node_num_training * args.test_increase_ratio)
    node_num_test_decrease = int(node_num_training * args.test_decrease_ratio)
    node_training = node_order[:node_num_training]
    node_test = np.concatenate(
        [
            node_order[: node_num_training - node_num_test_decrease],
            node_order[node_num_training : node_num_training + node_num_test_increase],
        ]
    )
    return {
        "node_training": node_training.astype(np.int64, copy=False),
        "node_test": node_test.astype(np.int64, copy=False),
        "node_sood_num": node_num_test_increase,
    }


def build_stone_blocks(args):
    s_block_num = int(getattr(args, "stone_SBlocks_num", 2))
    t_block_num = int(getattr(args, "stone_TBlocks_num", 5))
    s_blocks = [[64, 64, 64] for _ in range(s_block_num)]
    t_blocks = [[args.input_dim, 128, 128]]
    for _ in range(max(0, t_block_num - 1)):
        t_blocks.append([128, 128, 128])
    has_shallow_encode = [True] + [False] * max(0, s_block_num - 1)
    return (s_blocks, t_blocks, has_shallow_encode)


def compute_stone_runtime_split(node_runtime_num, args, node_sood_num, for_infer):
    if for_infer:
        node_num_un = int(node_sood_num)
    else:
        sample_ratio = float(getattr(args, "stone_num_sample", 0.01))
        node_num_un = max(1, int(node_runtime_num * sample_ratio))
    node_num_un = max(1, min(node_num_un, max(1, node_runtime_num - 1)))
    node_num_ob = node_runtime_num - node_num_un
    return (node_num_ob, node_num_un)


def compute_stone_sem_dim(node_num_ob, target_dim=None):
    if target_dim is not None:
        return int(target_dim)
    return max(1, int(math.ceil(math.log2(max(2, int(node_num_ob))))))


def build_model(
    model_name,
    args,
    node_total_num,
    node_runtime_num,
    adj_sub,
    node_sood_num=0,
    for_infer=False,
    stone_sem_dim=None,
):
    node_num_ob, node_num_un = compute_stone_runtime_split(
        node_runtime_num, args, node_sood_num=node_sood_num, for_infer=for_infer
    )
    sem_dim = compute_stone_sem_dim(node_num_ob, target_dim=stone_sem_dim)
    s_blocks, t_blocks, has_shallow_encode = build_stone_blocks(args)
    return STONE(
        SBlocks=s_blocks,
        TBlocks=t_blocks,
        node_num_un=node_num_un,
        node_num_ob=node_num_ob,
        sem_dim=sem_dim,
        has_shallow_encode=has_shallow_encode,
        Kt=int(getattr(args, "stone_Kt", 3)),
        Ks_s=int(getattr(args, "stone_Ks_s", 1)),
        Ks_t=int(getattr(args, "stone_Ks_t", 3)),
        dropout=float(getattr(args, "dropout", 0.3)),
        adp_s_dim=20,
        adp_t_dim=20,
        x_output_dim=128,
        sem_output_dim=64,
        gate_output_dim=128,
        node_num=node_runtime_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
    )
    raise ValueError("Unsupported model: {}".format(model_name))


def override_eval_args(base_args, dataset, device, scenario):
    values = vars(base_args).copy()
    values.update(
        {
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


def build_infer_model(
    model_name, args, node_total_num, node_eval_num, infer_adj, node_sood_num
):
    train_node_num = int(node_total_num / (1.0 + float(args.max_increase_ratio)))
    train_ob, _ = compute_stone_runtime_split(
        train_node_num, args, node_sood_num=node_sood_num, for_infer=False
    )
    stone_sem_dim = compute_stone_sem_dim(train_ob)
    return build_model(
        "stone",
        args,
        node_total_num,
        node_eval_num,
        infer_adj,
        node_sood_num=node_sood_num,
        for_infer=True,
        stone_sem_dim=stone_sem_dim,
    )
    raise ValueError("Unsupported model: {}".format(model_name))


def build_context(
    model_name, args, test_nodes, infer_adj, node_sood_num, device, node_total_num
):
    node_num_ob = len(test_nodes) - node_sood_num
    train_node_num = int(node_total_num / (1.0 + float(args.max_increase_ratio)))
    train_ob, _ = compute_stone_runtime_split(
        train_node_num, args, node_sood_num=node_sood_num, for_infer=False
    )
    sem_dim = compute_stone_sem_dim(train_ob)
    sem = compute_spatial_embedding(
        infer_adj.copy(),
        node_num_ob,
        c=int(getattr(args, "stone_c", 1)),
        seed=int(args.seed),
        device=str(device),
        target_dim=sem_dim,
    )
    return {"sem": sem}
    raise ValueError("Unsupported model: {}".format(model_name))


def forward_model(model_name, model, x, context):
    sem = context["sem"]
    if sem.dim() == 2:
        sem = sem.unsqueeze(0).expand(x.shape[0], -1, -1)
    pred_ob, pred_un, _ = model(x, sem)
    if pred_ob.dim() == 5:
        pred_ob = pred_ob.squeeze(0)
        pred_un = pred_un.squeeze(0)
    return torch.cat([pred_ob, pred_un], dim=2)
    raise ValueError("Unsupported model: {}".format(model_name))


def build_eval_split(node_total_num, args):
    split = build_split(node_total_num, args)
    train_nodes = split["node_training"]
    if int(args.sood):
        return (train_nodes, split["node_test"], int(split["node_sood_num"]))
    return (train_nodes, train_nodes, 0)


def load_checkpoint(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


def model_args(dataset):
    """Construct the architecture associated with the released checkpoint."""
    values = {
        "checkall": 0,
        "checkyears": "2018",
        "dropout": 0.3,
        "horizon": 12,
        "input_dim": 3,
        "max_increase_ratio": 0.3,
        "mode": "train",
        "output_dim": 1,
        "seed": 3028,
        "seq_len": 12,
        "sood": 1,
        "stone_KK": 2,
        "stone_Ks_s": 1,
        "stone_Ks_t": 3,
        "stone_Kt": 3,
        "stone_SBlocks_num": 2,
        "stone_TBlocks_num": 5,
        "stone_c": 1,
        "stone_gamma": 0.95,
        "stone_num_sample": 0.01,
        "stone_step_size": 10,
        "test_decrease_ratio": 0.1,
        "test_increase_ratio": 0.3,
        "tood": 1,
        "years": "2017",
    }
    values.update(
        {"SD": {"bs": 64, "dataset": "SD"}, "GBA": {"bs": 32, "dataset": "GBA"}}[
            dataset
        ]
    )
    return SimpleNamespace(**values)
