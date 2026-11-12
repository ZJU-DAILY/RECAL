import json
import numpy as np
import torch
from .base_scaler import BaseScaler


class SampleFirstZScoreScaler(BaseScaler):
    """
    Z-Score scaler that first splits samples (using idx files) and then
    computes mean / std only on the training slice.
    """

    def __init__(
        self,
        dataset_name: str,
        train_ratio: float,
        norm_each_channel: bool,
        rescale: bool,
        input_len: int,
        output_len: int,
    ):
        super().__init__(dataset_name, train_ratio, norm_each_channel, rescale)
        self.norm_channel = 0
        slice_dir = f"/{input_len}_{output_len}"
        meta_path = f"datasets/{dataset_name}/desc.json"
        with open(meta_path) as f:
            json.load(f)
        data_root = f"datasets/{dataset_name}"
        series = np.load(f"{data_root}/his.npz")["data"]
        train_indices = np.load(f"{data_root}{slice_dir}/idx_train.npy")
        last_train_idx = int(train_indices[-1])
        train_slice = series[: last_train_idx + 1, :, self.norm_channel]
        if norm_each_channel:
            mu = train_slice.mean(axis=0, keepdims=True)
            sigma = train_slice.std(axis=0, keepdims=True)
            sigma[sigma == 0] = 1.0
        else:
            mu = train_slice.mean()
            sigma = train_slice.std() or 1.0
        self.mu = torch.tensor(mu, dtype=torch.float32)
        self.sigma = torch.tensor(sigma, dtype=torch.float32)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        mu, sigma = (self.mu.to(x.device), self.sigma.to(x.device))
        x = x.clone()
        x[..., self.norm_channel] = (x[..., self.norm_channel] - mu) / sigma
        return x

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        mu, sigma = (self.mu.to(x.device), self.sigma.to(x.device))
        x = x.clone()
        x[..., self.norm_channel] = x[..., self.norm_channel] * sigma + mu
        return x
