import sys, argparse, random, torch

sys.path.append("src/")
import numpy as np
import os.path as osp
import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph
from src.model.model import STKEC_Model, EAC_Model
from dataer.SpatioTemporalDataset import SpatioTemporalDataset
from model import detect_default
from src.model import replay
from utils.initialize import init, seed_anything, init_log
from utils.common_tools import (
    mkdirs,
    load_best_model,
    long_term_pattern,
    load_test_best_model,
)
from trainer.default_trainer import test_model_with_recal
from recal import (
    DEFAULT_VERIFY_HALFLIFE_DAYS,
    FEEDBACK_MODES,
    LEGACY_FEEDBACK,
    REQUEST_KEYS,
    estimate_ticks_per_day,
)

TEST_MODES = ("recal",)


def normalize_test_mode(mode):
    if mode not in TEST_MODES:
        raise ValueError(f"Unsupported test_mode: {mode}. Use one of {TEST_MODES}.")
    return mode


def c3_source(args, inputs, year):
    """Pre-test series and stream alignment of one year for the C3 calendar memory (strict_causal only)."""
    if getattr(args, "recal_memory", "off") == "off" or args.test_mode != "recal":
        return None
    series = np.load(osp.join(args.raw_data_path, str(year) + ".npz"))["x"]
    if "PEMS" in args.raw_data_path:
        series = series[0 : 31 * 288, :]
    t = series.shape[0]
    first = int(t * 0.8)
    ticks = args.recal_memory_ticks_per_day or estimate_ticks_per_day(series[:first])
    return dict(
        series=series,
        train_span=slice(0, int(t * 0.2)),
        test_first_input_step=first,
        test_y=inputs["test_y"],
        ticks_per_day=int(ticks),
    )


def run_test_mode(model, args, test_loader):
    test_model_with_recal(
        model,
        args,
        test_loader,
        pin_memory=True,
        recal_rank=args.recal_rank,
        recal_lr=args.recal_lr,
        target_offset=args.recal_target_offset,
        max_batches=args.recal_max_batches,
        update_interval=args.recal_update_interval,
        maintenance_budget=args.recal_maintenance_budget,
        execution_backend=getattr(args, "recal_execution_backend", "reference"),
        feedback_mode=getattr(args, "recal_feedback_mode", LEGACY_FEEDBACK),
        allowed_lateness=getattr(args, "recal_allowed_lateness", 0),
        feedback_trace=getattr(args, "recal_feedback_trace", None),
    )


def main(args):
    args.logger.info("params : %s", vars(args))
    args.result = {
        "3": {" MAE": {}, "MAPE": {}, "RMSE": {}},
        "6": {" MAE": {}, "MAPE": {}, "RMSE": {}},
        "12": {" MAE": {}, "MAPE": {}, "RMSE": {}},
        "Avg": {" MAE": {}, "MAPE": {}, "RMSE": {}},
    }
    mkdirs(args.save_data_path)
    vars(args)["graph_size_list"] = []
    for year in range(args.begin_year, args.end_year + 1):
        graph = nx.from_numpy_array(
            np.load(osp.join(args.graph_path, str(year) + "_adj.npz"))["x"]
        )
        if year == args.begin_year:
            vars(args)["init_graph_size"] = graph.number_of_nodes()
            vars(args)["subgraph"] = graph
            vars(args)["base_node_size"] = graph.number_of_nodes()
        else:
            vars(args)["init_graph_size"] = args.graph_size
        vars(args)["graph_size"] = graph.number_of_nodes()
        vars(args)["year"] = year
        args.graph_size_list.append(graph.number_of_nodes())
        inputs = np.load(
            osp.join(args.save_data_path, str(year) + ".npz"), allow_pickle=True
        )
        args.logger.info(
            "[*] Year {} load from {}.npz".format(
                args.year, osp.join(args.save_data_path, str(year))
            )
        )
        adj = np.load(osp.join(args.graph_path, str(args.year) + "_adj.npz"))["x"]
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-06)
        vars(args)["adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
        if year == args.begin_year and args.load_first_year:
            model, _ = load_test_best_model(args)
            test_loader = DataLoader(
                SpatioTemporalDataset(inputs, "test"),
                batch_size=args.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=0,
            )
            vars(args)["c3_source"] = c3_source(args, inputs, year)
            run_test_mode(model, args, test_loader)
            continue
        vars(args)["node_list"] = list()
        if year > args.begin_year and args.strategy == "incremental":
            vars(args)["init_graph_size"] = np.load(
                osp.join(args.graph_path, str(year - 1) + "_adj.npz")
            )["x"].shape[0]
            model, _ = load_test_best_model(args)
            node_list = list()
            if args.increase:
                cur_node_size = np.load(
                    osp.join(args.graph_path, str(year) + "_adj.npz")
                )["x"].shape[0]
                pre_node_size = np.load(
                    osp.join(args.graph_path, str(year - 1) + "_adj.npz")
                )["x"].shape[0]
                node_list.extend(list(range(pre_node_size, cur_node_size)))
            pre_data = np.load(osp.join(args.raw_data_path, str(year - 1) + ".npz"))[
                "x"
            ][0 : 31 * 288, :]
            cur_data = np.load(osp.join(args.raw_data_path, str(year) + ".npz"))["x"][
                0 : 31 * 288, :
            ]
            pre_graph = np.array(
                list(
                    nx.from_numpy_array(
                        np.load(osp.join(args.graph_path, str(year - 1) + "_adj.npz"))[
                            "x"
                        ]
                    ).edges
                )
            ).T
            cur_graph = np.array(
                list(
                    nx.from_numpy_array(
                        np.load(osp.join(args.graph_path, str(year) + "_adj.npz"))["x"]
                    ).edges
                )
            ).T
            topk = int(0.05 * args.graph_size)
            if args.detect or args.replay:
                influence_node_score = detect_default.influence_node_selection(
                    model, args, pre_data, cur_data, pre_graph, cur_graph
                )
            if args.detect:
                influence_node_list = np.argpartition(
                    np.asarray(influence_node_score), -topk
                )[-topk:]
                node_list.extend(list(influence_node_list))
            if args.replay:
                replay_node_list = replay.replay_node_selection(
                    args, influence_node_score, topk
                )
                node_list.extend(list(replay_node_list))
            node_list = list(set(node_list))
            cur_graph = torch.LongTensor(
                np.array(
                    list(
                        nx.from_numpy_array(
                            np.load(osp.join(args.graph_path, str(year) + "_adj.npz"))[
                                "x"
                            ]
                        ).edges
                    )
                ).T
            )
            edge_list = list(
                nx.from_numpy_array(
                    np.load(osp.join(args.graph_path, str(year) + "_adj.npz"))["x"]
                ).edges
            )
            graph_node_from_edge = set()
            for u, v in edge_list:
                graph_node_from_edge.add(u)
                graph_node_from_edge.add(v)
            node_list = list(set(node_list) & graph_node_from_edge)
            "\n            If the node list is not empty\n            Returns the original graph node set, the original graph edge index, and the new index of the node set used for query in the subgraph (central node set) of num_hops hops. \n            Since relabel_nodes is set to True, the nodes will be relabeled from 0, so the original graph edge index is changed to the index of the new graph.\n            "
            if len(node_list) != 0:
                subgraph, subgraph_edge_index, mapping, _ = k_hop_subgraph(
                    node_list,
                    num_hops=args.num_hops,
                    edge_index=cur_graph,
                    relabel_nodes=True,
                )
                vars(args)["subgraph"] = subgraph
                vars(args)["subgraph_edge_index"] = subgraph_edge_index
                vars(args)["mapping"] = mapping
            args.logger.info(
                "number of increase nodes:{}, nodes after {} hop:{}, total nodes this year {}".format(
                    len(node_list), args.num_hops, args.subgraph.size(), args.graph_size
                )
            )
            vars(args)["node_list"] = np.asarray(node_list)
        if (
            args.strategy != "retrain"
            and year > args.begin_year
            and (len(args.node_list) == 0)
        ):
            raise ValueError(
                "No incremental nodes: stage checkpoint selection needs explicit review."
            )
        if args.train:
            raise ValueError(
                "Use externally trained stage checkpoints; this entry evaluates RECAL only."
            )
        elif args.auto_test:
            model, _ = load_test_best_model(args)
            test_loader = DataLoader(
                SpatioTemporalDataset(inputs, "test"),
                batch_size=args.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=0,
            )
            vars(args)["c3_source"] = c3_source(args, inputs, year)
            run_test_mode(model, args, test_loader)
    args.logger.info("\n\n")
    for i in ["3", "6", "12", "Avg"]:
        for j in [" MAE", "RMSE", "MAPE"]:
            info = ""
            info_list = []
            for year in range(args.begin_year, args.end_year + 1):
                if i in args.result:
                    if j in args.result[i]:
                        if year in args.result[i][j]:
                            info += "{:>10.2f}\t".format(args.result[i][j][year])
                            info_list.append(args.result[i][j][year])
            args.logger.info(
                "{:<4}\t{}\t".format(i, j)
                + info
                + "\t{:>8.2f}".format(np.mean(info_list))
            )
    total_time = 0
    for year in range(args.begin_year, args.end_year + 1):
        if year in args.result:
            info = "year\t{:<4}\ttotal_time\t{:>10.4f}\taverage_time\t{:>10.4f}\tepoch\t{}".format(
                year,
                args.result[year]["total_time"],
                args.result[year]["average_time"],
                args.result[year]["epoch_num"],
            )
            total_time += args.result[year]["total_time"]
            args.logger.info(info)
    args.logger.info("total time: {:.4f}".format(total_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--conf", type=str, default="conf/test.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paral", type=int, default=0)
    parser.add_argument("--gpuid", type=int, default=2)
    parser.add_argument("--logname", type=str, default="info")
    parser.add_argument("--method", type=str, default="EAC")
    parser.add_argument(
        "--load_first_year",
        type=int,
        default=1,
        help="0: training first year, 1: load from model path of first year",
    )
    parser.add_argument(
        "--first_year_model_path",
        type=str,
        default="log/ENERGY-Wind/eac-42/0/2.88.pkl",
        help="specify a pretrained model root",
    )
    parser.add_argument("--test_mode", type=str, default="recal", choices=TEST_MODES)
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
        help="RECAL execution only; hardware_aware is opt-in, independent of feedback policy",
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
        "--recal_memory", type=str, default="auto_multi", choices=REQUEST_KEYS
    )
    parser.add_argument(
        "--recal_memory_ticks_per_day",
        type=int,
        default=0,
        help="0 = estimate from the pre-test series (PEMS 288, ENERGY-Wind 144)",
    )
    parser.add_argument("--recal_memory_decay", type=float, default=0.9)
    parser.add_argument("--recal_memory_shrink", type=float, default=4.0)
    parser.add_argument("--recal_memory_root_shrink", type=float, default=None)
    parser.add_argument("--recal_memory_joint", type=int, default=1, choices=(0, 1))
    parser.add_argument("--recal_memory_verify", type=int, default=1, choices=(0, 1))
    parser.add_argument(
        "--recal_memory_verify_halflife_days",
        type=float,
        default=DEFAULT_VERIFY_HALFLIFE_DAYS,
    )
    parser.add_argument("--recal_memory_slot_ticks", type=int, default=1)
    parser.add_argument("--recal_memory_null_tol", type=float, default=0.0)
    parser.add_argument("--recal_memory_weekend", type=str, default="")
    args = parser.parse_args()
    args.test_mode = normalize_test_mode(args.test_mode)
    vars(args)["device"] = (
        torch.device("cuda:{}".format(args.gpuid))
        if torch.cuda.is_available() and args.gpuid != -1
        else "cpu"
    )
    vars(args)["methods"] = {"STKEC": STKEC_Model, "EAC": EAC_Model}
    init(args)
    seed_anything(args.seed)
    init_log(args)
    main(args)
