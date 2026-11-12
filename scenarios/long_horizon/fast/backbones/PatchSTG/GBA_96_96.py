import os
import sys
import torch
from easydict import EasyDict
from basicts.scaler import SampleFirstZScoreScaler
from basicts.utils import get_regular_settings
from .arch import PatchSTG
from .arch import reorderData

DATA_NAME = "GBA"
num_nodes = 2352
INPUT_LEN = 96
OUTPUT_LEN = 96
NUM_EPOCHS = 50
BATCH_SIZE = 64
OPTIM = "AdamW"
spa_patchsize = 2
spa_patchnum = 2048
factors = 128
node_dims = 32
recur = 11
metapath = f"./datasets/{DATA_NAME}/{DATA_NAME}_meta.csv"
adjpath = f"./datasets/{DATA_NAME}/adj_mx.pkl"
ori_parts_idx, reo_parts_idx, reo_all_idx = reorderData(
    metapath, adjpath, recur, spa_patchsize
)
regular_settings = get_regular_settings(DATA_NAME)
TRAIN_VAL_TEST_RATIO = regular_settings["TRAIN_VAL_TEST_RATIO"]
NORM_EACH_CHANNEL = regular_settings["NORM_EACH_CHANNEL"]
RESCALE = regular_settings["RESCALE"]
NULL_VAL = regular_settings["NULL_VAL"]
MODEL_ARCH = PatchSTG
MODEL_PARAM = {
    "tem_patchsize": INPUT_LEN,
    "tem_patchnum": 1,
    "output_len": OUTPUT_LEN,
    "node_num": num_nodes,
    "spa_patchsize": spa_patchsize,
    "spa_patchnum": spa_patchnum,
    "tod": 96,
    "dow": 7,
    "layers": 5,
    "factors": factors,
    "input_dims": 128,
    "node_dims": node_dims,
    "tod_dims": 32,
    "dow_dims": 32,
    "ori_parts_idx": ori_parts_idx,
    "reo_parts_idx": reo_parts_idx,
    "reo_all_idx": reo_all_idx,
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
CFG.METRICS.NULL_VAL = NULL_VAL
