import os, re, json, torch
import os.path as osp
import numpy as np
from Bio.Cluster import kcluster


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_json_file(file_path):
    with open(file_path, encoding="utf-8") as handle:
        return json.load(handle)


def load_best_model(args):
    if args.load_first_year:
        load_path = args.first_year_model_path
        loss = load_path.split("/")[-1].replace(".pkl", "")
    else:
        loss = []
        for filename in os.listdir(
            osp.join(
                args.model_path, args.logname + "-" + str(args.seed), str(args.year - 1)
            )
        ):
            if filename.endswith(".pkl"):
                loss.append(filename[0:-4])
        loss = sorted(loss)
        load_path = osp.join(
            args.model_path,
            args.logname + "-" + str(args.seed),
            str(args.year - 1),
            loss[0] + ".pkl",
        )
    args.logger.info("[*] load from {}".format(load_path))
    state_dict = torch.load(load_path, map_location=args.device)["model_state_dict"]
    model = args.methods[args.method](args)
    if args.method == "EAC":
        if args.year == args.begin_year:
            model.expand_adaptive_params(args.base_node_size)
        else:
            for idx, _ in enumerate(range(args.year - args.begin_year)):
                model.expand_adaptive_params(args.graph_size_list[idx])
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    return (model, loss[0])


def load_test_best_model(args):
    loss = []
    for filename in os.listdir(
        osp.join(os.environ["RECAL_STAGE_CHECKPOINT_ROOT"], str(args.year))
    ):
        if filename.endswith(".pkl"):
            loss.append(filename[0:-4])
    loss = sorted(loss)
    load_path = osp.join(
        os.environ["RECAL_STAGE_CHECKPOINT_ROOT"], str(args.year), loss[0] + ".pkl"
    )
    args.logger.info("[*] load from {}".format(load_path))
    state_dict = torch.load(load_path, map_location=args.device)["model_state_dict"]
    model = args.methods[args.method](args)
    if args.method == "EAC":
        if args.year == args.begin_year:
            model.expand_adaptive_params(args.base_node_size)
        else:
            for idx, _ in enumerate(range(args.year - args.begin_year)):
                model.expand_adaptive_params(args.graph_size_list[idx + 1])
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    return (model, loss[0])


def long_term_pattern(args, long_pattern):
    attention, _, _ = kcluster(long_pattern, nclusters=args.cluster, dist="u")
    np_attention = np.zeros((len(attention), args.cluster))
    for i in attention:
        np_attention[i][attention[i]] = 1.0
    return np_attention.astype(np.float32)


def get_max_columns(matrix):
    tensor = torch.tensor(matrix)
    max_columns, _ = torch.max(tensor, dim=1)
    return max_columns
