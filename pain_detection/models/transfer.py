"""
Cross-context transfer learning: pretrain on PMED, fine-tune on PMCD.

Strategies:
  - frozen:    only classifier head + fusion layer are updated
  - finetune:  all parameters updated, encoder lr reduced by 10x
"""
import numpy as np
from .attention_fusion import AttentionFusionModel


class TransferModel:
    """Wraps AttentionFusionModel with a two-stage transfer learning pipeline.

    Parameters
    ----------
    n_modalities   : number of shared modalities (e.g. 4 for BVP/EDA/EMG/Resp)
    n_classes_src  : classes in source task (PMED, e.g. 2 for binary)
    n_classes_tgt  : classes in target task (PMCD, e.g. 3)
    strategy       : 'frozen' or 'finetune'
    lr_factor      : learning rate reduction factor for encoders in finetune mode
    **kwargs       : passed to AttentionFusionModel (latent_dim, filters, etc.)
    """

    def __init__(
        self,
        n_modalities: int,
        n_classes_src: int,
        n_classes_tgt: int,
        strategy: str = "finetune",
        lr_factor: float = 0.1,
        **kwargs,
    ):
        self.n_modalities = n_modalities
        self.n_classes_src = n_classes_src
        self.n_classes_tgt = n_classes_tgt
        self.strategy = strategy
        self.lr_factor = lr_factor
        self.kwargs = kwargs

        # Will be set after pretrain
        self._encoder_state = None
        self._model = None  # the fine-tuned model used for predict

    def pretrain(self, X_src: np.ndarray, y_src: np.ndarray):
        """Stage 1: train on source domain (PMED)."""
        src_model = AttentionFusionModel(
            n_modalities=self.n_modalities,
            n_classes=self.n_classes_src,
            **self.kwargs,
        )
        src_model.fit(X_src, y_src)
        self._encoder_state = src_model.get_encoder_state()
        # Save normalisation stats from source for reference
        self._src_mean = src_model._mean
        self._src_std = src_model._std
        return self

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Stage 2: fine-tune on target domain (PMCD).

        If pretrain() was not called, trains from scratch (baseline comparison).
        """
        import torch

        self._model = AttentionFusionModel(
            n_modalities=self.n_modalities,
            n_classes=self.n_classes_tgt,
            **self.kwargs,
        )

        if self._encoder_state is not None:
            freeze = (self.strategy == "frozen")
            self._model.load_encoder_state(self._encoder_state, freeze=freeze)

        if self._encoder_state is not None and self.strategy == "finetune":
            # Use differential learning rates: encoder lr * lr_factor
            self._fit_differential_lr(X, y)
        else:
            # Frozen or from-scratch: standard training
            self._model.fit(X, y)

        return self

    def _fit_differential_lr(self, X: np.ndarray, y: np.ndarray):
        """Fine-tune with lower lr for encoders, normal lr for head."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        model = self._model

        X_t = np.transpose(X, (0, 2, 1)).astype(np.float32)
        model._mean = X_t.mean(axis=(0, 2), keepdims=True)
        model._std  = X_t.std(axis=(0, 2), keepdims=True) + 1e-8
        X_t = (X_t - model._mean) / model._std
        y_t = y.astype(np.int64)

        ds = TensorDataset(torch.tensor(X_t), torch.tensor(y_t))
        loader = DataLoader(ds, batch_size=model.batch_size, shuffle=True)

        # Differential lr: encoder params get lr * lr_factor
        encoder_params = list(model.encoders.parameters()) + \
                         list(model.projections.parameters())
        fusion_params  = list(model.W_a.parameters()) + list(model.w.parameters())
        head_params    = list(model.classifier.parameters())

        opt = torch.optim.Adam([
            {"params": encoder_params, "lr": model.lr * self.lr_factor},
            {"params": fusion_params,  "lr": model.lr * self.lr_factor},
            {"params": head_params,    "lr": model.lr},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=model.epochs)

        model._set_train(True)
        for epoch in range(model.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(model.device), yb.to(model.device)
                opt.zero_grad()
                logits, _ = model._forward(xb)
                loss = model._compute_loss(logits, yb)
                loss.backward()
                opt.step()
            scheduler.step()
        model._set_train(False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        return self._model.get_attention_weights(X)
