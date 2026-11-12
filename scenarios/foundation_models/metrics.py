import numpy as np


class Metrics:

    def __init__(self, horizon, nodes):
        self.absolute = np.zeros((horizon, nodes), np.float64)
        self.square = np.zeros_like(self.absolute)
        self.percentage = np.zeros_like(self.absolute)
        self.count = np.zeros_like(self.absolute)

    def add(self, pred, truth):
        assert pred.shape == truth.shape == self.absolute.shape
        if not np.isfinite(pred).all() or not np.isfinite(truth).all():
            raise ValueError("Non-finite metric input")
        valid = np.abs(truth) > 0.01
        residual = pred.astype(np.float64) - truth
        self.absolute += np.where(valid, np.abs(residual), 0)
        self.square += np.where(valid, residual**2, 0)
        self.percentage += np.divide(
            np.abs(residual), np.abs(truth), out=np.zeros_like(residual), where=valid
        )
        self.count += valid

    def result(self, nodes=None):
        nodes = slice(None) if nodes is None else nodes
        c = self.count[:, nodes].sum(axis=1)
        if np.any(c == 0):
            raise ValueError("No valid labels for a horizon")
        mae = self.absolute[:, nodes].sum(axis=1) / c
        rmse = np.sqrt(self.square[:, nodes].sum(axis=1) / c)
        mape = self.percentage[:, nodes].sum(axis=1) / c
        return dict(
            mae=float(mae.mean()),
            rmse=float(rmse.mean()),
            mape=float(mape.mean()),
            mape_scale="ratio",
            by_horizon=[
                dict(horizon=i + 1, mae=float(a), rmse=float(r), mape=float(m))
                for i, (a, r, m) in enumerate(zip(mae, rmse, mape))
            ],
        )
