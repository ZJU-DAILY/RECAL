import os
import sys
import torch
from easydict import EasyDict
from basicts.scaler import SampleFirstZScoreScaler
from basicts.utils import get_regular_settings, load_adj
from .arch import BigST
import pdb

DATA_NAME = "SD"
num_nodes = 716
INPUT_LEN = 96
OUTPUT_LEN = 96
NUM_EPOCHS = 50
BATCH_SIZE = 64
regular_settings = get_regular_settings(DATA_NAME)
TRAIN_VAL_TEST_RATIO = regular_settings["TRAIN_VAL_TEST_RATIO"]
NORM_EACH_CHANNEL = regular_settings["NORM_EACH_CHANNEL"]
RESCALE = regular_settings["RESCALE"]
NULL_VAL = regular_settings["NULL_VAL"]
PREPROCESSED_FILE = ""
MODEL_ARCH = BigST
adj_mx, _ = load_adj("datasets/" + DATA_NAME + "/adj_mx.pkl", "doubletransition")
MODEL_PARAM = {
    "bigst_args": {
        "num_nodes": num_nodes,
        "seq_num": INPUT_LEN,
        "in_dim": 3,
        "out_dim": OUTPUT_LEN,
        "hid_dim": 32,
        "tau": 0.25,
        "random_feature_dim": 64,
        "node_emb_dim": 32,
        "time_emb_dim": 32,
        "use_residual": True,
        "use_bn": True,
        "use_long": False,
        "use_spatial": True,
        "dropout": 0.3,
        "supports": [torch.tensor(i) for i in adj_mx],
        "time_of_day_size": 96,
        "day_of_week_size": 7,
    },
    "preprocess_path": PREPROCESSED_FILE,
    "preprocess_args": {
        "num_nodes": num_nodes,
        "in_dim": 3,
        "dropout": 0.3,
        "input_length": INPUT_LEN,
        "output_length": OUTPUT_LEN,
        "nhid": 32,
        "tiny_batch_size": 700,
    },
}
CFG = EasyDict()
CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = SampleFirstZScoreScaler
CFG.SCALER.PARAM = EasyDict(
    {
        "dataset_name": DATA_NAME,
        "train_ratio": TRAIN_VAL_TEST_RATIO[0],
        "norm_each_channel": NORM_EACH_CHANNEL,
        "rescale": RESCALE,
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
    }
)
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.METRICS = EasyDict()
CFG.METRICS.NULL_VAL = NULL_VAL
