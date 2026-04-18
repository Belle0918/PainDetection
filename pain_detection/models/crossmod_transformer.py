"""
CrossMod-Transformer for multimodal pain detection.

Adapted from Farmani et al. (Scientific Reports 2025, s41598-025-14238-y):
"A CrossMod-Transformer deep learning framework for multi-modal pain detection
through EDA and ECG fusion".

Architecture (per modality):
  1. FCN branch:  Conv1d(128,k=8) → Conv1d(256,k=5) → Conv1d(128,k=3) → GAP → 128-d
  2. ALSTM branch: LSTM(hidden=64, 2 layers) + Bahdanau attention (ELiSH) →
     concat(context, last_hidden) → 128-d
  3. Uni-modal Transformer: two TransformerEncoder blocks (8 heads, d=128) operate
     on FCN tokens and ALSTM tokens; two cross-attention blocks let FCN/ALSTM
     features attend to each other. Concat → mean-pool → FC(512→256→128).

Multi-modal fusion (CrossMod):
  Stack the 128-d uni-modal features of all M modalities as a length-M token
  sequence and apply a Transformer encoder + cross-attention between modality
  pairs, followed by FC(512→256→128→n_classes). This is the M>=2 generalisation
  of the paper's 2-modality cross-attention.

All modules are end-to-end trainable (unlike the paper's multi-step schedule,
which mainly helped with very long per-block epoch budgets). Focal loss and
class weights are supported for the 3-class / imbalanced settings.

Input: (N, T, C) — C channels treated as C modalities (1 channel each).
"""
from __future__ import annotations
import numpy as np


class CrossModTransformer:
    """End-to-end CrossMod-Transformer.

    Compatible with the fit/predict/predict_proba interface used by loso_cv.

    Parameters
    ----------
    n_modalities : number of modality channels (each fed as a univariate stream)
    n_classes    : number of output classes
    d_model      : latent dimension shared across branches (paper: 128)
    fcn_filters  : filter widths for the 3 conv layers (paper: 128, 256, 128)
    fcn_kernels  : kernel sizes for the 3 conv layers (paper: 8, 5, 3)
    lstm_hidden  : LSTM hidden size (paper: 64, 2 layers, bidirectional sums)
    lstm_layers  : number of stacked LSTM layers
    n_heads      : attention heads in each Transformer / cross-attention block
    ffn_hidden   : feed-forward hidden size in Transformer encoders
    dropout      : dropout rate applied inside blocks
    lr           : Adam learning rate
    epochs       : training epochs
    batch_size   : mini-batch size
    focal_loss   : use focal loss instead of cross-entropy
    focal_gamma  : focal loss focusing parameter
    class_weights: per-class loss weights (n_classes,) or None
    weight_decay : AdamW weight decay
    warmup_epochs: linear warmup length for cosine schedule
    target_length: downsample each window to this length via linear interp (paper
                   used 138 timesteps for 5.5s @ 25Hz); None = no resampling
    device       : 'cuda' / 'cpu' (auto-detect if None)
    random_state : seed
    """

    def __init__(
        self,
        n_modalities: int,
        n_classes: int,
        d_model: int = 128,
        fcn_filters: tuple[int, ...] = (128, 256, 128),
        fcn_kernels: tuple[int, ...] = (7, 5, 3),
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        n_heads: int = 8,
        ffn_hidden: int = 128,
        dropout: float = 0.3,
        lr: float = 5e-4,
        epochs: int = 80,
        batch_size: int = 64,
        focal_loss: bool = False,
        focal_gamma: float = 2.0,
        class_weights: np.ndarray | None = None,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        target_length: int | None = 250,
        device: str | None = None,
        random_state: int = 42,
    ):
        import torch

        torch.manual_seed(random_state)
        np.random.seed(random_state)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.n_modalities = n_modalities
        self.n_classes = n_classes
        self.d_model = d_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.focal_loss = focal_loss
        self.focal_gamma = focal_gamma
        self.target_length = target_length

        # Build the network
        self.net = _CrossModNet(
            n_modalities=n_modalities,
            n_classes=n_classes,
            d_model=d_model,
            fcn_filters=fcn_filters,
            fcn_kernels=fcn_kernels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            n_heads=n_heads,
            ffn_hidden=ffn_hidden,
            dropout=dropout,
        ).to(self.device)

        if class_weights is not None:
            self._class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        else:
            self._class_weights = None

    # ─── Training ─────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray):
        """X: (N, T, C) raw windows. y: (N,) int labels."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # (N, T, C) → (N, C, T) : M modalities × T timesteps
        X_t = np.transpose(X, (0, 2, 1)).astype(np.float32)
        X_t = self._resample(X_t)

        # Per-modality robust scaling (median / IQR) — matches the paper's RobustScaler
        med = np.median(X_t, axis=(0, 2), keepdims=True)
        q1  = np.quantile(X_t, 0.25, axis=(0, 2), keepdims=True)
        q3  = np.quantile(X_t, 0.75, axis=(0, 2), keepdims=True)
        iqr = np.maximum(q3 - q1, 1e-6)
        self._median = med
        self._iqr = iqr
        X_t = (X_t - med) / iqr

        # Auto class weights if focal_loss not set and classes are imbalanced
        if self._class_weights is None and not self.focal_loss:
            counts = np.bincount(y.astype(int), minlength=self.n_classes).astype(np.float32)
            if counts.min() > 0 and counts.max() / counts.min() > 1.5:
                w = counts.sum() / (self.n_classes * counts)
                self._class_weights = torch.tensor(w, dtype=torch.float32).to(self.device)

        y_t = y.astype(np.int64)
        ds = TensorDataset(torch.tensor(X_t), torch.tensor(y_t))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        opt = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        total_steps = max(1, len(loader)) * self.epochs
        warmup_steps = max(1, len(loader)) * self.warmup_epochs

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        self.net.train()
        step = 0
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad()
                logits = self.net(xb)
                loss = self._loss(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                opt.step()
                scheduler.step()
                step += 1
        self.net.eval()
        return self

    def _loss(self, logits, targets):
        import torch
        import torch.nn.functional as F
        if self.focal_loss:
            ce = F.cross_entropy(logits, targets, weight=self._class_weights, reduction="none")
            pt = torch.exp(-ce)
            return ((1 - pt) ** self.focal_gamma * ce).mean()
        return F.cross_entropy(logits, targets, weight=self._class_weights)

    # ─── Inference ────────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        x = self._prep(X)
        with torch.no_grad():
            logits = self.net(x)
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        x = self._prep(X)
        with torch.no_grad():
            logits = self.net(x)
            proba = F.softmax(logits, dim=1)
        return proba.cpu().numpy()

    def _prep(self, X: np.ndarray):
        import torch
        X_t = np.transpose(X, (0, 2, 1)).astype(np.float32)
        X_t = self._resample(X_t)
        X_t = (X_t - self._median) / self._iqr
        return torch.tensor(X_t, dtype=torch.float32).to(self.device)

    def _resample(self, X_t: np.ndarray) -> np.ndarray:
        """Linearly interpolate each (N, M, T) window to target_length along T."""
        if self.target_length is None or X_t.shape[-1] == self.target_length:
            return X_t
        T_in = X_t.shape[-1]
        T_out = self.target_length
        xp = np.linspace(0.0, 1.0, T_in, dtype=np.float32)
        xq = np.linspace(0.0, 1.0, T_out, dtype=np.float32)
        N, M, _ = X_t.shape
        out = np.empty((N, M, T_out), dtype=np.float32)
        for n in range(N):
            for m in range(M):
                out[n, m] = np.interp(xq, xp, X_t[n, m])
        return out


# ──────────────────────────────────────────────────────────────────────────────
#  Internal PyTorch modules
# ──────────────────────────────────────────────────────────────────────────────
def _build_net():
    """Factory guard used only for readability — modules are defined lazily below."""
    return None


class _ELiSH:
    """ELiSH activation used in the Bahdanau attention score layer.
       ELiSH(x) = x for x>=0, else (e^x - 1) / (1 + e^{-x})"""
    def __call__(self, x):
        import torch
        pos = x
        neg = (torch.expm1(x)) / (1.0 + torch.exp(-x))
        return torch.where(x >= 0, pos, neg)


def _fcn_block(in_ch, filters, kernels, dropout):
    import torch.nn as nn
    layers = []
    c_in = in_ch
    for f, k in zip(filters, kernels):
        layers += [
            nn.Conv1d(c_in, f, kernel_size=k, padding=k // 2),
            nn.BatchNorm1d(f),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        c_in = f
    return nn.Sequential(*layers)


class _ALSTM:
    pass


def _crossmod_import_check():
    # Deferred torch import to keep `import pain_detection` cheap.
    import torch  # noqa: F401
    return True


# Actual nn.Module classes need torch at import time; define them via a factory.
def _make_net_cls():
    import torch
    import torch.nn as nn

    class ALSTMBranch(nn.Module):
        """LSTM + Bahdanau attention with ELiSH score.
        Input:  (B, 1, T)
        Output: tokens (B, T, d_model), pooled 128-d feature (B, d_model)
        """
        def __init__(self, d_model: int, hidden: int, layers: int, dropout: float):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1, hidden_size=hidden, num_layers=layers,
                batch_first=True, dropout=dropout if layers > 1 else 0.0,
            )
            # Attention: score_t = w^T ELiSH(h_t)  (paper eq. 7)
            self.score = nn.Linear(hidden, 1, bias=False)
            self.elish = _ELiSH()
            # Project concat([context, h_last]) ∈ R^{2*hidden} → d_model
            self.proj = nn.Linear(2 * hidden, d_model)
            # Token projection for cross-attention: each timestep hidden → d_model
            self.tok_proj = nn.Linear(hidden, d_model)

        def forward(self, x):
            # x: (B, 1, T) → (B, T, 1)
            x = x.transpose(1, 2)
            h, (h_n, _) = self.lstm(x)              # h: (B, T, H)
            e = self.score(self.elish(h))           # (B, T, 1)
            alpha = torch.softmax(e, dim=1)         # (B, T, 1)
            context = (alpha * h).sum(dim=1)        # (B, H)
            z = torch.cat([context, h_n[-1]], dim=-1)  # (B, 2H)
            pooled = self.proj(z)                   # (B, d)
            tokens = self.tok_proj(h)               # (B, T, d) — for cross-attn
            return tokens, pooled

    class FCNBranch(nn.Module):
        """3-layer FCN. Returns per-timestep tokens AND GAP-pooled vector."""
        def __init__(self, d_model: int, filters, kernels, dropout: float):
            super().__init__()
            self.net = _fcn_block(1, filters, kernels, dropout)
            # Project per-timestep feature map channel → d_model for cross-attn
            self.tok_proj = nn.Linear(filters[-1], d_model)
            self.pool_proj = nn.Linear(filters[-1], d_model) if filters[-1] != d_model else nn.Identity()

        def forward(self, x):
            # x: (B, 1, T)
            feat = self.net(x)              # (B, F, T)
            tokens = self.tok_proj(feat.transpose(1, 2))   # (B, T, d)
            pooled = self.pool_proj(feat.mean(dim=-1))     # (B, d)   ← GAP
            return tokens, pooled

    class TransformerEncoderBlock(nn.Module):
        def __init__(self, d_model: int, n_heads: int, ffn_hidden: int, dropout: float):
            super().__init__()
            self.layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=ffn_hidden,
                dropout=dropout, activation="relu", batch_first=True, norm_first=False,
            )

        def forward(self, x):
            return self.layer(x)

    class CrossAttention(nn.Module):
        """Cross-attention: Q from stream A, K/V from stream B.
        Post-attention: residual + LayerNorm + FFN + LayerNorm (like encoder)."""
        def __init__(self, d_model: int, n_heads: int, ffn_hidden: int, dropout: float):
            super().__init__()
            self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.ln1 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, ffn_hidden), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(ffn_hidden, d_model),
            )
            self.ln2 = nn.LayerNorm(d_model)
            self.drop = nn.Dropout(dropout)

        def forward(self, q_src, kv_src):
            a, _ = self.mha(q_src, kv_src, kv_src, need_weights=False)
            x = self.ln1(q_src + self.drop(a))
            x = self.ln2(x + self.drop(self.ffn(x)))
            return x

    class UniModalBranch(nn.Module):
        """FCN + ALSTM + intra-modal Transformer + cross-attention → 128-d feat."""
        def __init__(self, d_model, fcn_filters, fcn_kernels, lstm_hidden,
                     lstm_layers, n_heads, ffn_hidden, dropout):
            super().__init__()
            self.fcn  = FCNBranch(d_model, fcn_filters, fcn_kernels, dropout)
            self.alstm = ALSTMBranch(d_model, lstm_hidden, lstm_layers, dropout)
            self.te_fcn  = TransformerEncoderBlock(d_model, n_heads, ffn_hidden, dropout)
            self.te_alstm = TransformerEncoderBlock(d_model, n_heads, ffn_hidden, dropout)
            self.ca_fcn_to_alstm = CrossAttention(d_model, n_heads, ffn_hidden, dropout)
            self.ca_alstm_to_fcn = CrossAttention(d_model, n_heads, ffn_hidden, dropout)
            # Mean-pool across time of concatenated streams → (B, 2d) → FC1 → FC2 → FC3 (d_model)
            self.fc = nn.Sequential(
                nn.Linear(2 * d_model, 512), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, d_model), nn.ReLU(),
            )

        def forward(self, x):
            # x: (B, 1, T)
            fcn_tok, _  = self.fcn(x)          # (B, T_f, d)
            als_tok, _  = self.alstm(x)        # (B, T_a, d)
            # Even-kernel convs can off-by-one the sequence length relative to
            # the LSTM output; align to the shorter one so cross-attn is valid.
            T_min = min(fcn_tok.size(1), als_tok.size(1))
            fcn_tok = fcn_tok[:, :T_min]
            als_tok = als_tok[:, :T_min]
            f = self.te_fcn(fcn_tok)           # intra-modal encoder on FCN stream
            l = self.te_alstm(als_tok)         # intra-modal encoder on ALSTM stream
            f2 = self.ca_fcn_to_alstm(f, l)    # FCN attends to ALSTM
            l2 = self.ca_alstm_to_fcn(l, f)    # ALSTM attends to FCN
            cat = torch.cat([f2, l2], dim=-1)  # (B, T, 2d)
            pooled = cat.mean(dim=1)           # (B, 2d)
            return self.fc(pooled)             # (B, d_model)

    class CrossModNet(nn.Module):
        def __init__(self, n_modalities, n_classes, d_model, fcn_filters, fcn_kernels,
                     lstm_hidden, lstm_layers, n_heads, ffn_hidden, dropout):
            super().__init__()
            self.uni = nn.ModuleList([
                UniModalBranch(d_model, fcn_filters, fcn_kernels, lstm_hidden,
                               lstm_layers, n_heads, ffn_hidden, dropout)
                for _ in range(n_modalities)
            ])
            # Learnable modality embedding appended to each modality token
            self.mod_emb = nn.Parameter(torch.randn(n_modalities, d_model) * 0.02)
            # Multi-modal Transformer operates on a length-M token sequence
            self.mm_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=ffn_hidden,
                    dropout=dropout, activation="relu", batch_first=True, norm_first=False,
                ),
                num_layers=2,
            )
            # Classification head: FC4(512)→FC5(256)→FC6(128)→FC7(n_classes)
            self.head = nn.Sequential(
                nn.Linear(d_model, 512), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            # x: (B, M, T)
            feats = []
            for m, branch in enumerate(self.uni):
                xm = x[:, m:m+1, :]                   # (B, 1, T)
                feats.append(branch(xm))              # (B, d)
            H = torch.stack(feats, dim=1)             # (B, M, d)
            H = H + self.mod_emb.unsqueeze(0)         # + modality embedding
            H = self.mm_encoder(H)                    # inter-modal attention
            z = H.mean(dim=1)                         # (B, d) — pool over modalities
            return self.head(z)

    return CrossModNet


# Lazy construction: build the nn.Module class on first use so `import
# pain_detection.models` does not force-import torch for non-DL users.
def _CrossModNet(**kwargs):
    cls = _make_net_cls()
    return cls(**kwargs)
