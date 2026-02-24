from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _MovingAverage(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        self.kernel_size = int(kernel_size)
        self.pool = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        if self.kernel_size == 1:
            return x
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        x_3d = x.unsqueeze(1)  # (B, 1, L)
        left = x_3d[:, :, :1].repeat(1, 1, pad_left) if pad_left > 0 else x_3d[:, :, :0]
        right = x_3d[:, :, -1:].repeat(1, 1, pad_right) if pad_right > 0 else x_3d[:, :, :0]
        padded = torch.cat([left, x_3d, right], dim=2)
        return self.pool(padded).squeeze(1)


class DLinear(nn.Module):
    """Minimal univariate DLinear / LTSF-Linear style model."""

    def __init__(self, lookback: int, horizon: int, moving_avg: int = 25) -> None:
        super().__init__()
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        if self.lookback < 1 or self.horizon < 1:
            raise ValueError("lookback and horizon must be >= 1")
        ma_kernel = max(1, min(int(moving_avg), self.lookback))
        self.moving_avg = _MovingAverage(ma_kernel)
        self.linear_seasonal = nn.Linear(self.lookback, self.horizon)
        self.linear_trend = nn.Linear(self.lookback, self.horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        trend = self.moving_avg(x)
        seasonal = x - trend
        return self.linear_seasonal(seasonal) + self.linear_trend(trend)


def _as_float32_arrays(X, Y) -> tuple[np.ndarray, np.ndarray]:
    X_arr = np.asarray(X, dtype=np.float32)
    Y_arr = np.asarray(Y, dtype=np.float32)
    if X_arr.ndim != 2 or Y_arr.ndim != 2:
        raise ValueError("X and Y must be 2D arrays")
    if X_arr.shape[0] != Y_arr.shape[0]:
        raise ValueError("X and Y must have the same number of rows")
    return X_arr, Y_arr


def train_dlinear(
    X_train,
    Y_train,
    X_val,
    Y_val,
    *,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 42,
) -> DLinear:
    """Train a univariate DLinear model with MSE loss and simple early stopping."""
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if lr <= 0:
        raise ValueError("lr must be > 0")

    X_tr, Y_tr = _as_float32_arrays(X_train, Y_train)
    X_va, Y_va = _as_float32_arrays(X_val, Y_val)
    if X_tr.shape[1] != X_va.shape[1] or Y_tr.shape[1] != Y_va.shape[1]:
        raise ValueError("Train/val dimensions must match")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    model = DLinear(lookback=X_tr.shape[1], horizon=Y_tr.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=False, drop_last=False)

    X_val_t = torch.from_numpy(X_va).to(device)
    Y_val_t = torch.from_numpy(Y_va).to(device)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    patience = 5
    patience_left = patience

    for _epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = float(loss_fn(val_pred, Y_val_t).cpu().item())

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_dlinear(model: DLinear, X) -> np.ndarray:
    """Predict multi-horizon outputs from a trained DLinear model."""
    X_arr = np.asarray(X, dtype=np.float32)
    if X_arr.ndim != 2:
        raise ValueError("X must be 2D")
    if X_arr.shape[0] == 0:
        return np.empty((0, model.horizon), dtype=float)

    device = next(model.parameters()).device
    with torch.no_grad():
        pred = model(torch.from_numpy(X_arr).to(device)).cpu().numpy()
    return np.asarray(pred, dtype=float)

