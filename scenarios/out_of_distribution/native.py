"""2017-trained STONE / REACT with RECAL on the joint spatial-temporal OOD stream."""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import torch
from adapters import stone, react
from data import build_node_splits, get_dataset_info
from stream import OODStream, load_test_stream
from calendar_source import calendar_memory_kwargs


class StoneStream(OODStream):
    def forecast(self, x):
        return stone.forward_model("stone", self.model, x, self.context)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("stone", "react"), required=True)
    p.add_argument("--dataset", choices=("SD", "GBA"), required=True)
    p.add_argument("--year", required=True)
    p.add_argument("--new-node-ratio", type=float, required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--max-steps", type=int, default=0)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("recal.ood")
    torch.set_num_threads(3)
    device = torch.device(a.device)
    scenario = dict(checkyears=a.year, inc=a.new_node_ratio, dec=0.1, sood=1, tood=1)
    if a.model == "stone":
        args = stone.override_eval_args(
            stone.model_args(a.dataset), a.dataset, a.device, scenario
        )
    else:
        args = react.override_eval_args(
            react.model_args(a.dataset), a.dataset, a.device, scenario, a.data_root
        )
    args.bs = 1
    data_path, adj_path, n, _ = get_dataset_info(a.dataset, a.data_root)
    react.set_seed(args.seed)
    if a.model == "stone":
        train, nodes, unseen = stone.build_eval_split(n, args)
        adjacency = np.load(adj_path)[np.ix_(nodes, nodes)].copy()
        np.fill_diagonal(adjacency, 0)
        react.set_seed(args.seed)
        model = stone.build_infer_model(
            "stone", args, n, len(nodes), adjacency, unseen
        ).to(device)
        stone.load_checkpoint(model, a.checkpoint, device)
        context = stone.build_context(
            "stone", args, nodes, adjacency, unseen, device, n
        )
    else:
        split = build_node_splits(n, args, log)
        train, nodes, unseen = (
            split["node_training"],
            split["node_test"],
            split["node_unseen_num"],
        )
        graph = react.load_graph_resources(a.dataset, args, log, {}, n, train)
        model = react.build_model(args, n, graph["static_prior"], graph["adj_mx"])
        state = torch.load(a.checkpoint, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    series, idx, scaler, _ = load_test_stream(data_path, args, nodes, 64)
    if len(idx) != 6976 or not np.all(np.diff(idx) == 1):
        raise ValueError("Paper OOD stream requires 6976 contiguous origins per year.")
    stream = (StoneStream if a.model == "stone" else OODStream)(
        model, series, idx, scaler, nodes, unseen, 12, 12, device, batch_size=1
    )
    if a.model == "stone":
        stream.context = context
    memory = calendar_memory_kwargs(
        data_path,
        "2017",
        a.year,
        train,
        len(idx),
        12,
        12,
        "auto_multi",
        target_offset=1,
        logger=log,
        null_tol=0.01,
    )
    result = stream.run_recal(
        rank=4,
        lr=1e-4,
        target_offset=1,
        seed=a.seed,
        backend="reference",
        max_batches=a.max_steps,
        **memory
    )
    result.update(
        model=a.model,
        dataset=a.dataset,
        train_year=2017,
        test_year=a.year,
        new_node_ratio=a.new_node_ratio,
        training_seed=args.seed,
        calibration_seed=a.seed,
        is_exclusive=False,
        efficiency_usable=False,
    )
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
