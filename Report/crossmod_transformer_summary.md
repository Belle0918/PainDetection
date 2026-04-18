# CrossMod-Transformer for Multimodal Pain Detection

## Method

We adapted the **CrossMod-Transformer** architecture from Farmani et al. (*Scientific Reports*, 2025) to our PMED / PMCD datasets. The model combines classical time-series feature extractors with a two-stage Transformer fusion scheme.

### Architecture

For each physiological modality (BVP, EDA, Respiration, EMG), a dedicated branch produces a 128-dimensional representation:

1. **FCN branch** — 3 Conv1d layers (128 → 256 → 128 filters, kernel sizes 7 / 5 / 3) + BatchNorm + ReLU + Dropout, followed by a token projection for subsequent cross-attention.
2. **ALSTM branch** — 2-layer LSTM (hidden size 64) with **Bahdanau-style attention** using the ELiSH activation from the paper, yielding a context vector concatenated with the last hidden state.
3. **Intra-modal Transformer** — Two `TransformerEncoder` blocks (8 heads, d=128, FFN hidden=128) operate on the FCN and ALSTM token streams, followed by two **cross-attention blocks** where each stream uses the other as keys/values. Features are concatenated, mean-pooled, and passed through FC(512 → 256 → 128).

For **inter-modal fusion (CrossMod)**, the 128-d uni-modal features from all modalities are stacked as a length-M token sequence, augmented with a learnable **modality embedding**, and processed by a 2-layer Transformer encoder. The pooled output is fed through FC(512 → 256 → 128 → n_classes). This generalises the paper's 2-modality design (EDA + ECG) to M ≥ 2 modalities.

### Training

- **Optimiser**: AdamW, weight decay 1e-4
- **Schedule**: 5-epoch linear warmup + cosine decay
- **Gradient clipping**: max norm 1.0
- **Pre-processing**: linear downsampling to 250 timesteps, per-modality RobustScaler (median / IQR)
- **Class imbalance**: automatic inverse-frequency class weights when imbalance > 1.5×
- **Loss**: cross-entropy (focal loss supported but not used in final runs)
- **Framework**: PyTorch, end-to-end trainable (the paper's multi-step schedule was simplified)

### Evaluation

5-fold subject-grouped cross-validation (no subject appears in both train and test). All metrics are macro-averaged over all held-out samples.

## Results

### PMCD (Clinical Pain, 3-class: No-pain / Moderate / Severe)

| Model | Macro F1 | Accuracy | Macro AUC |
|---|---|---|---|
| **CrossMod-Transformer (ours)** | **0.594** | **0.694** | **0.786** |
| Qwen3-4B + LoRA + clf_head (r5) | 0.464 | 0.494 | 0.655 |

**Improvement of +0.13 macro F1 over a 4B-parameter LLM**, with a much smaller model (~4.6M parameters).

Per-class F1 on PMCD:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No-pain | 0.75 | 0.81 | 0.78 | 1443 |
| Moderate | 0.70 | 0.73 | 0.72 | 1527 |
| Severe | 0.36 | 0.23 | 0.28 | 485 |

### PMED (Experimental Heat Pain, 3-class, CoVAS labels)

| Model | Macro F1 | Accuracy | Macro AUC |
|---|---|---|---|
| CrossMod-Transformer (ours) | 0.416 | 0.458 | 0.629 |
| Qwen3-4B + LoRA + clf_head (r5) | **0.649** | **0.858** | **0.940** |

On PMED, the LLM-based approach clearly outperforms our CrossMod-Transformer.

## Key Findings

1. **Clinical data favours specialised fusion architectures.** On PMCD — where pain labels reflect real clinical episodes with heterogeneous, noisy physiological responses — the inductive bias of the FCN + ALSTM + cross-attention design captures complementary patterns that a general-purpose LLM misses.

2. **Severe-class imbalance remains a bottleneck on PMCD.** The minority class (485 / 3455 samples) achieves only 0.28 F1 despite automatic class weighting; further focal-loss / oversampling experiments did not break the ceiling within our training budget.

3. **PMED's inter-subject variability is the primary obstacle.** Experimental heat stimulation produces physiological responses that vary dramatically between subjects for the same stimulus intensity. A per-sample robust-scaling variant was tried to remove amplitude differences but destroyed pain-relevant information on PMCD while not meaningfully helping on PMED — we reverted it.

4. **LLMs generalise across datasets more robustly.** The teammate's Qwen3-4B clf_head achieves F1 = 0.65 on PMED partly because the large pre-trained backbone encodes representations that are less sensitive to subject-specific signal idiosyncrasies. A specialised fusion model must compensate for this with stronger domain-specific regularisation and — arguably — per-subject calibration data that our current setup does not use.

5. **Parameter efficiency.** Our model (4.6M parameters) reaches competitive or superior performance to a 4B-parameter LLM (≈ 1000× smaller) on PMCD, which is relevant for real-time or embedded deployment scenarios.

## Limitations and Future Work

- The per-subject RobustScaler used by the original CrossMod-Transformer paper requires labelled baseline samples from each test subject. Implementing this in a proper LOSO setting (using only training-subject statistics plus a small held-out calibration set per test subject) is a natural next step and may close the PMED gap.
- Data augmentation (time warping, jittering, channel dropout) has not yet been explored and is known to benefit physiological-signal models.
- The paper's multi-step training schedule (pre-training each sub-network independently before end-to-end fusion) was simplified in our implementation; reverting to the original schedule may improve PMED performance at the cost of additional training time.

## References

Farmani, J., Bargshady, G., Gkikas, S., Tsiknakis, M., & Fernandez Rojas, R. (2025). A CrossMod-Transformer deep learning framework for multi-modal pain detection through EDA and ECG fusion. *Scientific Reports*, 15(1), 29467. https://doi.org/10.1038/s41598-025-14238-y
