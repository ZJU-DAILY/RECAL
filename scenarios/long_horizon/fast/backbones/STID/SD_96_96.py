import os
import sys
import torch
from easydict import EasyDict
from basicts.scaler import SampleFirstZScoreScaler
from basicts.utils import get_regular_settings, load_adj
from .arch import STID

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
MODEL_ARCH = STID
MODEL_PARAM = {
    "num_nodes": num_nodes,
    "input_len": INPUT_LEN,
    "input_dim": 1,
    "embed_dim": 32,
    "output_len": OUTPUT_LEN,
    "num_layer": 4,
    "if_node": True,
    "node_dim": 64,
    "if_T_i_D": True,
    "if_D_i_W": True,
    "temp_dim_tid": 32,
    "temp_dim_diw": 32,
    "time_of_day_size": 96,
    "day_of_week_size": 7,
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
