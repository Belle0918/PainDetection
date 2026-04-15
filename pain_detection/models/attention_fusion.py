"""
Attention-Fusion Model for multimodal pain detection.

Architecture (from paper Section 4.3):
  1. Each modality is processed by an independent 1D-CNN temporal encoder.
  2. Modality representations are fused via learned attention weights.
  3. The fused representation is classified by a 2-layer MLP.

Input: (N, T, C) where C channels are split into M single-channel modalities.
Each modality encoder receives (N, 1, T) and outputs h^(m) ∈ R^d.
Attention computes α_m and produces z = Σ α_m · h^(m).
"""
import numpy as np


class AttentionFusionModel:
    """Modality-specific encoders with attention-based fusion.

    Compatible with the fit/predict/predict_proba interface used by loso_cv.

    Parameters
    ----------
    n_modalities : number of input channels (each treated as one modality)
    n_classes    : number of output classes
    latent_dim   : shared representation dimension d for each encoder
    filters      : conv filter sizes for each encoder block
    kernel_size  : conv kernel size
    dropout      : dropout rate
    lr           : learning rate
    epochs       : training epochs
    batch_size   : mini-batch size
    focal_loss   : if True, use focal loss instead of cross-entropy
    focal_gamma  : gamma parameter for focal loss
    class_weights: optional array of per-class weights for loss
    device       : 'cuda' or 'cpu'
    random_state : random seed
    """

    def __init__(
        self,
        n_modalities: int,
        n_classes: int,
        latent_dim: int = 64,
        filters: tuple[int, ...] = (32, 64),
        kernel_size: int = 7,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        focal_loss: bool = False,
        focal_gamma: float = 2.0,
        class_weights: np.ndarray | None = None,
        device: str | None = None,
        random_state: int = 42,
    ):
        import torch
        import torch.nn as nn

        torch.manual_seed(random_state)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_modalities = n_modalities
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.focal_loss = focal_loss
        self.focal_gamma = focal_gamma

        # ── Modality-specific encoders ────────────────────────────────────────
        self.encoders = nn.ModuleList()
        for _ in range(n_modalities):
            layers: list[nn.Module] = []
            in_ch = 1  # each modality is a single channel
            for out_ch in filters:
                layers += [
                    nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                              padding=kernel_size // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
                in_ch = out_ch
            layers.append(nn.AdaptiveAvgPool1d(1))  # → (N, filters[-1], 1)
            encoder = nn.Sequential(*layers)
            self.encoders.append(encoder)

        # Project each encoder output to shared latent_dim
        self.projections = nn.ModuleList([
            nn.Linear(filters[-1], latent_dim) for _ in range(n_modalities)
        ])

        # ── Attention fusion ──────────────────────────────────────────────────
        # α_m = softmax( w^T tanh(W_a h^(m) + b_a) )
        self.W_a = nn.Linear(latent_dim, latent_dim, bias=True)
        self.w = nn.Linear(latent_dim, 1, bias=False)

        # ── Classification head (2-layer MLP) ────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, n_classes),
        )

        # Move everything to device
        self.encoders = self.encoders.to(self.device)
        self.projections = self.projections.to(self.device)
        self.W_a = self.W_a.to(self.device)
        self.w = self.w.to(self.device)
        self.classifier = self.classifier.to(self.device)

        # Store class weights
        if class_weights is not None:
            self._class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        else:
            self._class_weights = None

    def _all_parameters(self):
        """Gather all trainable parameters."""
        import itertools
        return itertools.chain(
            self.encoders.parameters(),
            self.projections.parameters(),
            self.W_a.parameters(),
            self.w.parameters(),
            self.classifier.parameters(),
        )

    def _forward(self, x: "torch.Tensor"):
        """
        x : (N, M, T)  — M modalities, each single-channel with T timesteps.
        Returns logits (N, n_classes) and attention weights (N, M).
        """
        import torch

        N, M, T = x.shape
        h_list = []
        for m in range(M):
            x_m = x[:, m:m+1, :]          # (N, 1, T)
            feat = self.encoders[m](x_m)   # (N, filters[-1], 1)
            feat = feat.squeeze(-1)        # (N, filters[-1])
            h_m = self.projections[m](feat) # (N, latent_dim)
            h_list.append(h_m)

        # Stack: (N, M, latent_dim)
        H = torch.stack(h_list, dim=1)

        # Attention weights: (N, M)
        scores = self.w(torch.tanh(self.W_a(H)))  # (N, M, 1)
        alpha = torch.softmax(scores.squeeze(-1), dim=1)  # (N, M)

        # Weighted sum: z = Σ α_m · h^(m)  →  (N, latent_dim)
        z = (alpha.unsqueeze(-1) * H).sum(dim=1)

        logits = self.classifier(z)  # (N, n_classes)
        return logits, alpha

    def _compute_loss(self, logits, targets):
        """Cross-entropy or focal loss."""
        import torch
        import torch.nn.functional as F

        if self.focal_loss:
            ce = F.cross_entropy(logits, targets, weight=self._class_weights, reduction='none')
            pt = torch.exp(-ce)
            loss = ((1 - pt) ** self.focal_gamma * ce).mean()
        else:
            loss = F.cross_entropy(logits, targets, weight=self._class_weights)
        return loss

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X : (N, T, C) raw windows — C channels treated as C modalities
        y : (N,) integer class labels
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # (N, T, C) → (N, C, T)  i.e. (N, M, T)
        X_t = np.transpose(X, (0, 2, 1)).astype(np.float32)

        # Per-modality z-score normalisation
        self._mean = X_t.mean(axis=(0, 2), keepdims=True)  # (1, M, 1)
        self._std  = X_t.std(axis=(0, 2), keepdims=True) + 1e-8
        X_t = (X_t - self._mean) / self._std

        y_t = y.astype(np.int64)

        ds = TensorDataset(torch.tensor(X_t), torch.tensor(y_t))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(self._all_parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        self._set_train(True)
        for epoch in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                logits, _ = self._forward(xb)
                loss = self._compute_loss(logits, yb)
                loss.backward()
                opt.step()
            scheduler.step()
        self._set_train(False)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        X_t = self._prepare_input(X)
        with torch.no_grad():
            logits, _ = self._forward(X_t)
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        X_t = self._prepare_input(X)
        with torch.no_grad():
            logits, _ = self._forward(X_t)
            proba = F.softmax(logits, dim=1)
        return proba.cpu().numpy()

    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """Return attention weights α (N, M) for interpretability."""
        import torch
        X_t = self._prepare_input(X)
        with torch.no_grad():
            _, alpha = self._forward(X_t)
        return alpha.cpu().numpy()

    def _prepare_input(self, X: np.ndarray) -> "torch.Tensor":
        """Normalise and convert to tensor on device."""
        import torch
        X_t = np.transpose(X, (0, 2, 1)).astype(np.float32)
        X_t = (X_t - self._mean) / self._std
        return torch.tensor(X_t, dtype=torch.float32).to(self.device)

    def _set_train(self, mode: bool):
        self.encoders.train(mode)
        self.projections.train(mode)
        self.W_a.train(mode)
        self.w.train(mode)
        self.classifier.train(mode)

    def get_encoder_state(self) -> dict:
        """Export encoder + fusion weights for transfer learning."""
        return {
            "encoders": self.encoders.state_dict(),
            "projections": self.projections.state_dict(),
            "W_a": self.W_a.state_dict(),
            "w": self.w.state_dict(),
        }

    def load_encoder_state(self, state: dict, freeze: bool = False):
        """Load pretrained encoder + fusion weights (for transfer learning)."""
        self.encoders.load_state_dict(state["encoders"])
        self.projections.load_state_dict(state["projections"])
        self.W_a.load_state_dict(state["W_a"])
        self.w.load_state_dict(state["w"])
        if freeze:
            for p in self.encoders.parameters():
                p.requires_grad = False
            for p in self.projections.parameters():
                p.requires_grad = False
