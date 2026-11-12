import argparse
from recal import (
    DEFAULT_VERIFY_HALFLIFE_DAYS,
    FEEDBACK_MODES,
    LEGACY_FEEDBACK,
    REQUEST_KEYS,
)

EVAL_METHODS = ("recal",)


def normalize_eval_method(method):
    if method not in EVAL_METHODS:
        raise ValueError(
            f"Unsupported eval_method: {method}. Use one of {EVAL_METHODS}."
        )
    return method


def get_public_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--dataset", type=str, default="PEMS08")
    parser.add_argument(
        "--fewshot",
        type=int,
        default=5,
        help="create few-shot percent datasets, 100 means full-shot",
    )
    parser.add_argument(
        "--eval_method", type=str, default="recal", choices=EVAL_METHODS
    )
    parser.add_argument(
        "--eval_tag",
        type=str,
        default="",
        help="optional suffix for test log filenames; default keeps existing names",
    )
    parser.add_argument("--recal_rank", type=int, default=4)
    parser.add_argument("--recal_lr", type=float, default=0.0001)
    parser.add_argument("--recal_target_offset", type=int, default=1)
    parser.add_argument("--recal_max_batches", type=int, default=0)
    parser.add_argument(
        "--recal_update_interval",
        type=int,
        default=1,
        help="merge k matured diagonals into one update (k>=1; 1=eager)",
    )
    parser.add_argument(
        "--recal_maintenance_budget",
        type=int,
        default=0,
        help="update only Top-K nodes per step by U-loading importance (0=all nodes)",
    )
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
        help="fixed replay (legacy) or bounded asynchronous feedback (event_time)",
    )
    parser.add_argument(
        "--recal_allowed_lateness",
        type=int,
        default=0,
        help="event_time only: maximum label wait L in ticks",
    )
    parser.add_argument(
        "--recal_feedback_trace",
        type=str,
        default=None,
        help="event_time only: npz with an [E,N] arrival schedule (-1=never); omit for on-time (F0) delivery",
    )
    parser.add_argument(
        "--recal_memory",
        type=str,
        default="auto_multi",
        choices=REQUEST_KEYS,
        help="calendar memory key; auto = decided on pre-test data",
    )
    parser.add_argument("--recal_memory_decay", type=float, default=0.9)
    parser.add_argument("--recal_memory_shrink", type=float, default=4.0)
    parser.add_argument(
        "--recal_memory_root_shrink",
        type=float,
        default=None,
        help="pseudo-count toward 0 at the root level (hier keys); default = shrink",
    )
    parser.add_argument("--recal_memory_joint", type=int, default=1, choices=(0, 1))
    parser.add_argument("--recal_memory_verify", type=int, default=1, choices=(0, 1))
    parser.add_argument(
        "--recal_memory_verify_halflife_days",
        type=float,
        default=DEFAULT_VERIFY_HALFLIFE_DAYS,
    )
    parser.add_argument(
        "--recal_memory_slot_ticks",
        type=int,
        default=1,
        help="ticks merged into one time-of-day slot (3 = 15-min slots on 5-min data)",
    )
    parser.add_argument(
        "--recal_memory_null_tol",
        type=float,
        default=0.01,
        help="|label| <= tol counts as missing (zeros after inverse transform)",
    )
    parser.add_argument(
        "--recal_memory_weekend",
        type=str,
        default="",
        help="comma-separated weekend dow indices; empty = inferred",
    )
    parser.add_argument("--use_long", type=bool, default=False)
    parser.add_argument("--years", type=str, default="2016")
    parser.add_argument("--seed", type=int, default=2016)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--input_dim", type=int, default=3)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    return parser
