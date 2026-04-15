"""
Early-fusion 1D-CNN baseline.

All modalities are concatenated along the channel dimension and processed
by a single shared backbone.  Input shape: (N, T, C).
"""
import numpy as np


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class CNN1D:
    """Thin wrapper around a PyTorch 1D-CNN.

    The architecture uses 1D convolutions followed by global average pooling,
    making it agnostic to input length T.

    Input shape to fit/predict: (N, T, C)  — time-first, converted internally.
    """

    def __init__(
        self,
        n_sensors: int,
        n_classes: int,
        filters: tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 7,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        device: str | None = None,
        random_state: int = 42,
    ):
        if not _torch_available():
            raise ImportError("PyTorch is required for CNN1D.")

        import torch
        import torch.nn as nn

        torch.manual_seed(random_state)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_classes = n_classes

        layers: list[nn.Module] = []
        in_ch = n_sensors
        for out_ch in filters:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool1d(1))

        self.backbone = nn.Sequential(*layers).to(self.device)
        self.head = nn.Linear(filters[-1], n_classes).to(self.device)

    def _forward(self, x_np: np.ndarray):
        import torch
        x = torch.tensor(x_np, dtype=torch.float32).to(self.device)
        feat = self.backbone(x).squeeze(-1)
        return self.head(feat)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """X : (N, T, C) raw windows;  y : (N,) integer labels."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        X_t = np.transpose(X, (0, 2, 1)).astype(np.float32)
        y_t = y.astype(np.int64)

        self._mean = X_t.mean(axis=(0, 2), keepdims=True)
        self._std  = X_t.std(axis=(0, 2), keepdims=True) + 1e-8
        X_t = (X_t - self._mean) / self._std

        ds = TensorDataset(torch.tensor(X_t), torch.tensor(y_t))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        params = list(self.backbone.parameters()) + list(self.head.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                out = self.head(self.backbone(xb).squeeze(-1))
                loss = criterion(out, yb)
                loss.backward()
                opt.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        X_t = np.transpose(X, (0, 2, 1)).astype(np.float32)
        X_t = (X_t - self._mean) / self._std
        with torch.no_grad():
            logits = self._forward(X_t)
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        X_t = np.transpose(X, (0, 2, 1)).astype(np.float32)
        X_t = (X_t - self._mean) / self._std
        with torch.no_grad():
            logits = self._forward(X_t)
            proba  = F.softmax(logits, dim=1)
        return proba.cpu().numpy()
