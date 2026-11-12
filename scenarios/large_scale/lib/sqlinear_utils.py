import torch
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# log string
def log_string(log, string):
    log.write(string + "\n")
    log.flush()
    print(string)


# metric
def metric(pred, label):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        wape = np.divide(np.sum(mae), np.sum(label))
        wape = np.nan_to_num(wape * mask)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def _compute_loss(y_true, y_predicted):
    return masked_mae(y_predicted, y_true, 0.0)


def seq2instance(data, P, Q):
    num_step = data.shape[0]
    num_sample = num_step - P - Q + 1
    if num_sample <= 0:
        raise ValueError(
            f"Invalid sequence lengths: got {data.shape[0]} steps for P={P}, Q={Q}."
        )
    windows = sliding_window_view(data, P + Q, axis=0)
    windows = np.moveaxis(windows, -1, 1)
    x = np.ascontiguousarray(windows[:, :P])
    y = np.ascontiguousarray(windows[:, P:])
    return x, y


def read_meta(path):
    meta = pd.read_csv(path)
    # Keep float64: float32 rounding creates coordinate ties and changes the Square Partition order.
    lat = meta["Lat"].to_numpy(dtype=np.float64)
    lng = meta["Lng"].to_numpy(dtype=np.float64)
    locations = np.stack([lat, lng], 0)
    return locations


def reorderData(parts_idx):
    ori_parts_idx = []
    reo_parts_idx = []
    reo_all_idx = []

    current_offset = 0
    for part_idx in parts_idx:
        part_size = part_idx.shape[0]
        local_reo_idx = np.arange(
            current_offset, current_offset + part_size, dtype=np.int32
        )

        reo_parts_idx.append(local_reo_idx)
        ori_parts_idx.append(part_idx.astype(np.int32, copy=False))
        reo_all_idx.append(part_idx.astype(np.int32, copy=False))

        current_offset += part_size

    return (
        np.concatenate(ori_parts_idx),
        np.concatenate(reo_parts_idx),
        np.concatenate(reo_all_idx),
    )


def square_partition(points, C=8):
    """
    Square partition recursive partitioning.

    Args:
        points: numpy array of shape (N, 2), each row is [lng, lat]
        C: capacity per patch

    Returns:
        patches: list of arrays, each array contains indices of points in that patch
    """
    patches = []

    def recursive_split(indices):
        if len(indices) <= C:
            patches.append(indices)
            return

        sub_points = points[indices]
        min_xy = sub_points.min(axis=0)
        max_xy = sub_points.max(axis=0)
        span = max_xy - min_xy
        major_axis = np.argmax(span)

        sorted_idx = np.argsort(sub_points[:, major_axis])
        sorted_indices = indices[sorted_idx]

        N = len(sorted_indices)
        candidates = [i for i in range(C, N, C)]
        if not candidates:
            split_idx = N // 2
        else:
            mid = N // 2
            split_idx = min(candidates, key=lambda x: abs(x - mid))

        left = sorted_indices[:split_idx]
        right = sorted_indices[split_idx:]
        recursive_split(left)
        recursive_split(right)

    all_indices = np.arange(len(points))
    recursive_split(all_indices)
    return patches


def loadData(
    filepath, metapath, P, Q, train_ratio, test_ratio, adjpath, tod, dow, capacity, log
):
    # Traffic
    Traffic = np.load(filepath)["data"][..., :1].astype(np.float32, copy=False)
    locations = read_meta(metapath)
    num_step = Traffic.shape[0]
    # temporal positions
    TE = np.zeros([num_step, 2], dtype=np.int16)
    TE[:, 0] = np.arange(num_step, dtype=np.int32) % tod
    TE[:, 1] = (np.arange(num_step, dtype=np.int32) // tod) % dow
    log_string(log, f"Shape of data: {Traffic.shape}")
    log_string(log, f"Shape of locations: {locations.shape}")
    # train/val/test
    train_steps = round(train_ratio * num_step)
    test_steps = round(test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    trainData, trainTE = Traffic[:train_steps], TE[:train_steps]
    valData, valTE = (
        Traffic[train_steps : train_steps + val_steps],
        TE[train_steps : train_steps + val_steps],
    )
    testData, testTE = Traffic[-test_steps:], TE[-test_steps:]
    parts_idx = square_partition(locations.T, C=capacity)
    patch_lengths = [len(part) for part in parts_idx]
    ori_parts_idx, reo_parts_idx, reo_all_idx = reorderData(parts_idx)

    # X, Y
    trainX, trainY = seq2instance(trainData, P, Q)
    valX, valY = seq2instance(valData, P, Q)
    testX, testY = seq2instance(testData, P, Q)
    trainXTE, trainYTE = seq2instance(trainTE, P, Q)
    valXTE, valYTE = seq2instance(valTE, P, Q)
    testXTE, testYTE = seq2instance(testTE, P, Q)
    # normalization
    mean, std = float(np.mean(trainX)), float(np.std(trainX))
    # log
    log_string(log, f"Shape of Train: {trainY.shape}")
    log_string(log, f"Shape of Validation: {valY.shape}")
    log_string(log, f"Shape of Test: {testY.shape}")
    log_string(
        log,
        f"Patch count: {len(parts_idx)}, min size: {min(patch_lengths)}, max size: {max(patch_lengths)}",
    )
    log_string(log, f"Mean: {mean} & Std: {std}")

    return (
        trainX,
        trainY,
        trainXTE,
        trainYTE,
        valX,
        valY,
        valXTE,
        valYTE,
        testX,
        testY,
        testXTE,
        testYTE,
        mean,
        std,
        ori_parts_idx,
        reo_parts_idx,
        reo_all_idx,
    )
