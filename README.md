# Cross-Context Physiological Pain Recognition

Automated pain recognition from multimodal physiological signals, studying cross-context transfer from experimental (PMED) to clinical (PMCD) settings on the PainMonit Database. The project covers classical feature-based baselines, neural attention-fusion models, and LLM-based approaches (SFT, GRPO / SRPO, classification-head).

## Project Structure

```
PainDetection/
├── pain_detection/                # Core library
│   ├── config.py                  # Paths, sensor names, sampling rate
│   ├── data_loader.py             # PMED/PMCD loading + label mapping
│   ├── features.py                # 16 statistical/spectral features per sensor
│   ├── evaluate.py                # LOSO-CV, grouped K-fold CV, held-out eval
│   └── models/
│       ├── classical.py           # Random Forest, SVM baselines
│       ├── cnn.py                 # Early-fusion 1D-CNN baseline
│       ├── attention_fusion.py    # Modality-specific encoders + attention fusion
│       ├── crossmod_transformer.py# FCN + ALSTM + Transformer cross-attention fusion
│       ├── transfer.py            # PMED→PMCD transfer learning wrapper
│       └── llm_baseline.py        # LLM-based classification (Claude API)
│
├── experiments/                   # Experiment scripts
│   ├── run_baseline.py            # RF / SVM / CNN baselines with LOSO-CV
│   ├── run_cross_domain.py        # Cross-domain: train PMED, test PMCD
│   ├── run_attention_fusion.py    # Attention-fusion model with LOSO-CV
│   ├── run_crossmod_transformer.py# CrossMod-Transformer (paper-adapted) with LOSO-CV
│   ├── run_transfer.py            # Transfer learning comparison
│   └── run_llm_baseline.py        # LLM zero-shot / few-shot baseline
│
├── PMED/                          # Experimental dataset preprocessing
├── PMCD/                          # Clinical dataset preprocessing
└── Report/                        # NeurIPS-format paper
```

## Dataset

This project uses the **PainMonit Database** (Gouverneur et al., 2024):

- **PMED**: 52 healthy subjects, controlled heat pain, 6 sensors (BVP, EDA x2, ECG, EMG, Resp)
- **PMCD**: 49 clinical patients, physiotherapy pain, 7 sensors (BVP x2, EDA x2, EMG, Temp, Resp)

> Gouverneur et al., "An Experimental and Clinical Physiological Signal Dataset for Automated Pain Recognition", *Scientific Data*, 2024. https://doi.org/10.1038/s41597-024-03878-w

## Setup

```bash
pip install -r requirements.txt
```

Place raw datasets in `PMED/dataset/raw-data/` and `PMCD/dataset/raw-data/`, then generate numpy files:

```bash
cd PMED && python create_np_files.py && cd ..
cd PMCD && python create_np_files.py && cd ..
```

## Models

| Model | Description | Script |
|---|---|---|
| RF / SVM | Feature-based baselines (16 features/sensor) | `run_baseline.py` |
| 1D-CNN | Early-fusion neural baseline (all channels concatenated) | `run_baseline.py --model cnn` |
| **Attention-Fusion** | Modality-specific 1D-CNN encoders + learned attention fusion | `run_attention_fusion.py` |
| **CrossMod-Transformer** | Per-modality FCN + ALSTM + two-stage Transformer cross-attention fusion, adapted from Farmani et al. (*Sci. Reports*, 2025) | `run_crossmod_transformer.py` |
| **Transfer Learning** | Pretrain on PMED → fine-tune on PMCD (frozen / full) | `run_transfer.py` |
| LLM Baseline | Claude API zero-shot / few-shot on extracted features | `run_llm_baseline.py` |
| LLM SFT (Qwen3-4B) | LoRA fine-tune with CoT output | via LLaMA-Factory |
| LLM + GRPO / SRPO | RL on top of SFT | via LLaMA-Factory / custom trainer |
| **LLM + clf_head** | Frozen backbone + LoRA + linear classification head | `scripts/clf_head_train.py` |

## Experiments

All scripts are in `experiments/` and run from the project root:

```bash
# Classical baselines
python experiments/run_baseline.py --dataset pmed --scheme 3class --model rf
python experiments/run_baseline.py --dataset pmcd --scheme 3class --model rf

# Cross-domain (RF on PMED → test on PMCD)
python experiments/run_cross_domain.py --model rf --scheme 3class --also-within

# Attention-fusion model
python experiments/run_attention_fusion.py --dataset pmed --scheme binary
python experiments/run_attention_fusion.py --dataset pmcd --scheme 3class

# CrossMod-Transformer (recommended: 5-fold grouped CV for speed, LOSO for final reporting)
python experiments/run_crossmod_transformer.py --dataset pmcd --scheme 3class \
    --epochs 100 --dropout 0.2 --lr 3e-4 --cv kfold --kfolds 5
python experiments/run_crossmod_transformer.py --dataset pmed --scheme 3class --label covas \
    --epochs 80 --dropout 0.2 --lr 3e-4 --cv kfold --kfolds 5

# Transfer learning (scratch vs frozen vs finetune)
python experiments/run_transfer.py --scheme 3class

# Modality ablation (single sensor)
python experiments/run_attention_fusion.py --dataset pmed --scheme binary --sensors Bvp
python experiments/run_attention_fusion.py --dataset pmed --scheme binary --sensors Eda_E4

# LLM baseline (requires ANTHROPIC_API_KEY)
python experiments/run_llm_baseline.py --dataset pmcd --scheme 3class --compare --context
```

## Results — Classical and Neural Baselines

Random Forest uses the 16-feature-per-sensor pipeline with LOSO-CV. "Improved Attention-Fusion" is the multi-scale encoder (kernels 3 / 7 / 15 / 31) + temporal attention pooling + InstanceNorm variant, trained with label smoothing 0.1; the PMCD run additionally uses focal loss (γ=2) and a class-balanced sampler to handle the 3.1× imbalance (No-pain 1443 / Moderate 1527 / Severe 485). All runs are LOSO-CV on the 4 shared modalities (BVP, EDA_E4, EMG, Resp).

| Setting | Model | Accuracy | Macro F1 | AUC |
|---|---|---|---|---|
| PMED CoVAS 3-class | Random Forest | 0.563 | 0.555 | 0.730 |
| PMED CoVAS 3-class | Attention-Fusion | 0.3944 | 0.2980 | 0.4857 |
| PMED Heater 3-class | Random Forest | 0.509 | 0.506 | 0.705 |
| PMED Heater 3-class | Improved Attention-Fusion | 0.4461 | 0.4342 | 0.6300 |
| PMCD 3-class | Random Forest | 0.576 | 0.483 | 0.688 |
| PMCD 3-class | Improved Attention-Fusion (focal + balanced) | 0.4851 | 0.4279 | 0.6681 |
| Cross-domain PMED→PMCD | Random Forest | 0.388 | 0.366 | 0.552 |

> Chance level for 3-class is 33.3%. Cross-domain accuracy near chance confirms a large domain gap between experimental and clinical pain. On these four-modality LOSO splits, the deep models do *not* beat RF on Macro F1 — Improved Attention-Fusion trails RF by ~0.07 on PMED Heater and ~0.06 on PMCD. On PMCD the deep model does improve Severe recall (21 % vs. RF's near-zero under imbalance) but loses Moderate precision. PMED CoVAS is the hardest split — subjective ratings are much noisier than heater ground truth — and the vanilla attention-fusion variant collapses to near-chance.

## Results — CrossMod-Transformer

Paper-adapted hierarchical fusion: each modality is processed by a parallel FCN (Conv 128→256→128, kernels 7/5/3) and an ALSTM (LSTM-64 + Bahdanau attention with ELiSH) branch; two intra-modal `TransformerEncoder` blocks plus bidirectional cross-attention fuse FCN and ALSTM features per modality; a separate inter-modal Transformer attends across the M modality-level tokens (augmented with a learnable modality embedding) before a 4-layer MLP classifier. Training: AdamW + cosine decay with 5-epoch warmup, gradient clipping, auto inverse-frequency class weights, per-modality RobustScaler, 5-fold subject-grouped CV on the 4 shared modalities (BVP, EDA_E4, Resp, EMG).

| Setting | Model | Accuracy | Macro F1 | AUC |
|---|---|---|---|---|
| **PMCD 3-class** | **CrossMod-Transformer** | **0.6944** | **0.5936** | **0.7856** |
| PMED 3-class (CoVAS) | CrossMod-Transformer | 0.4577 | 0.4155 | 0.6290 |
| PMED binary (CoVAS) | CrossMod-Transformer | 0.5683 | 0.5123 | — |

### Why PMCD F1 = 0.59 — and Why It Is Hard to Push Higher

PMCD has strong *class imbalance*: the Severe class is only 485 / 3455 (14%). Per-class breakdown from the best PMCD run:

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| No-pain | 1443 | 0.75 | 0.81 | 0.78 |
| Moderate | 1527 | 0.70 | 0.73 | 0.72 |
| Severe | **485** | 0.36 | **0.23** | **0.28** |
| **Macro** | 3455 | 0.61 | 0.59 | **0.59** |

The Severe class *alone* drags the macro F1 down by ~0.15. Auto-computed inverse-frequency weights (2.4× for Severe) are not enough — most Severe windows are mistakenly classified as Moderate (260/485), because moderate and severe clinical pain are physiologically close along BVP/EDA/Resp features. Paths we tried and their effect:

- **Focal loss (γ = 2)**: same macro F1 (0.59), slight Severe-recall gain, but no headline shift.
- **Lower dropout (0.2 vs 0.3) + lower LR (3e-4 vs 5e-4)**: +0.01 macro F1 — the run reported above.
- **Per-sample RobustScaler** (instead of per-modality across the training set): *worse* on PMCD (F1 dropped to ~0.47) because it removes the subject-level EDA/BVP amplitude cues that actually carry pain information. Kept per-modality scaling.

### Why PMED F1 Is Low — and Why the LLM Wins There

PMED is experimental heat pain with 87 subjects. Two properties make it much harder for a specialised fusion model than PMCD:

1. **Inter-subject variability dominates the signal.** For the same stimulus level (e.g. P3), different subjects show very different BVP, EDA, EMG responses. With subject-grouped CV, the held-out subjects have amplitude profiles the model has never seen — and signal normalisation alone cannot fix this. We verified this by switching to per-sample RobustScaler (which removes subject-level amplitude differences): it did not recover F1 on PMED, confirming the problem is *shape* and *timing* variability, not just amplitude.
2. **Labels are loosely coupled to perception.** Using `label=heater` treats P1 / P2 as "Lower" and P3 / P4 as "Higher" regardless of the subject's real pain experience; subjects with high pain tolerance produce "Higher-pain" physiology that looks like others' "Lower-pain". We switched to `label=covas` (subjective 0–100 rating quartiles) and F1 barely moved — the noise is inherent to the protocol, not the label scheme.

This is exactly where a large pre-trained LLM has the upper hand: Qwen3-4B + clf_head (r5) reaches F1 = 0.65 on PMED because its backbone already encodes robust general-purpose representations that are less brittle to signal idiosyncrasies. A 4.6 M-parameter specialised CNN/LSTM/Transformer stack does not have that prior.

### How CrossMod-Transformer Compares to Other Baselines

| Setting | RF (16 feats) | Attention-Fusion | CrossMod-Transformer | Qwen3-4B + clf_head |
|---|---|---|---|---|
| PMCD 3-class (Macro F1) | 0.483 | 0.428 | **0.594** | 0.464 |
| PMED 3-class CoVAS (Macro F1) | 0.555 | 0.298 | 0.416 | **0.649** |

On the **clinical PMCD split** — arguably the more clinically relevant task — CrossMod-Transformer is the best model in this repository, beating every other baseline including the 4 B-parameter LLM (+0.13 F1 / ~1 000× fewer parameters). On the **experimental PMED split** the LLM is strongest; the small fusion model ranks between RF and the LLM. This gap is mainly a function of PMED's inter-subject noise, not an architectural limitation of CrossMod per se: the paper's original setup used per-subject RobustScaler with a calibration window from each test subject, which we cannot reproduce cleanly under fully subject-held-out evaluation.

## Results — LLM Fine-Tuning Experiments

All Qwen3-4B runs below are LoRA fine-tuning on top of `Qwen3-4B-Instruct`. Two dataset variants are compared, and they differ in several ways — not only in the output format:

| | **v1** (`dataset for sft/`) | **v2** (`dataset for sft v2/`) |
|---|---|---|
| Normalization | Global feature discretization (all subjects pooled); subject baselines collapsed | Subject-level **z-score** on raw signal + subject-level z-score on window features (`high`/`low` is always subject-relative) |
| Features | Basic statistics (amplitude, range, spectral regularity, tonic/phasic EDA, RMS/ZCR EMG, breathing rate, …) | v1 features + **HRV (RMSSD)**, **SCR count**, **pulse rate**, **EMG burst count**, **respiration-rate variability** |
| Input text | `level (raw_value)` per feature | `level (raw=…, z=…)` per feature — level, raw, and z-score all shown |
| PMED labels | From original dataset scheme | Re-derived: segment on Heater plateaus → window COVAS median → `heater ≤ 32 or COVAS < 0.5 → No-pain`, else split against subject's positive-COVAS median into `Moderate`/`Severe` |
| Sample count | PMCD train ≈ 2339 | PMCD train ≈ 1241 (different windowing / stride) |
| Output target | Single-word label (`No-pain` / `Moderate` / `Severe`) | Chain-of-thought reasoning ending in `Classification: <label>` |

The v2 rebuild is generated by `rebuild_dataset_cot.py` directly from `raw data/`; see `dataset_rebuild.md` for the full recipe.

| Run | Method | Dataset format | Eval split | Accuracy | Macro F1 | Bal. Acc | AUC | Notes |
|---|---|---|---|---|---|---|---|---|
| — | Qwen3-4B LoRA (no-CoT baseline) | v1 | PMED | 0.3776 | 0.3777 | — | — | Single-word label output |
| — | Qwen3-4B LoRA (no-CoT baseline) | v1 | PMCD | 0.5269 | **0.4736** | — | — | Single-word label output |
| — | Qwen3-4B LoRA (no-CoT baseline) | v1 | PMED→PMCD | — | 0.4357 | — | — | Combined-domain transfer |
| r1 | Qwen3-4B SFT (2-stage) | v2 CoT | PMED | — | 0.5199 | — | — | No-pain recall only 6 % |
| r1 | Qwen3-4B SFT (2-stage) | v2 CoT | PMCD | — | 0.3589 | 0.4120 | — | class-imbalanced |
| r3 | Qwen3-4B + GRPO | v2 CoT | PMCD | — | 0.3546 | 0.3645 | — | No-pain recall 6 % → 42 % |
| r4 | Qwen3-4B + SRPO | v2 CoT | PMCD | — | 0.3782 | 0.4049 | 0.5497 | Best generative; Severe recall = 48 % |
| **r5** | **Qwen3-4B + clf_head** | v2 CoT input | **PMED** | **0.8579** | **0.6486** | **0.6620** | **0.9399** | ✅ **Best overall** |
| **r5** | **Qwen3-4B + clf_head** | v2 CoT input | **PMCD** | **0.4939** | **0.4642** | **0.4573** | **0.6548** | ✅ No parse failure |
| r5t | Qwen3-4B + clf_head | v2 CoT input | PMED → PMCD | — | 0.4181 | 0.4468 | 0.6234 | Transfer weaker than in-domain |
| r7p | Phi-4-mini + clf_head | v2 CoT input | PMED | — | 0.5662 | 0.6600 | 0.9378 | AUC matches r5, Moderate recall = 0 |
| r7 | Phi-4-mini + clf_head | v2 CoT input | PMCD | — | 0.2516 | 0.3333 | 0.4879 | ❌ Collapsed to majority class |
| r7ct | Phi-4-mini + clf_head | v2 CoT input | PMED → PMCD | — | 0.1311 | 0.3249 | 0.4979 | Near-random transfer |

> **Note on the v1 vs v2 comparison.** The v1 baseline and the v2 runs are *not* a clean apples-to-apples comparison — v2 changes the normalization scheme, feature set, PMED label derivation, windowing, and output target all at once. The v1 baseline row marks the pre-project starting point, not a controlled ablation. The classification-head runs (r5, r5t, r7p, r7, r7ct) use only the v2 **input**; their label head is a linear classifier, not a text generator, so they bypass the CoT output entirely. Headline deltas vs the v1 baseline: PMED Macro F1 +0.27 (0.38 → 0.65), PMED Accuracy +0.48 (0.38 → 0.86); PMCD in-domain is roughly on par (−0.009 F1); PMED → PMCD transfer is within 0.02 F1.

### r5 Best-Model Per-Class Breakdown

**PMCD val (n = 330)** — Accuracy 0.494, BalAcc 0.515, AUC 0.653:

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| No-pain | 69 | 0.49 | 0.33 | 0.40 |
| Moderate | 200 | 0.74 | 0.48 | 0.58 |
| Severe | 61 | 0.29 | 0.74 | 0.42 |
| **Macro** | 330 | 0.51 | 0.52 | **0.46** |

**PMED val (n = 556)** — Accuracy 0.858, BalAcc 0.667, AUC 0.940:

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| No-pain | 400 | 1.00 | 0.98 | 0.99 |
| Moderate | 66 | 0.40 | 0.29 | 0.34 |
| Severe | 90 | 0.56 | 0.73 | 0.64 |
| **Macro** | 556 | 0.66 | 0.67 | **0.65** |

### Key Findings

1. **Classification head > generative SFT.** The frozen-backbone + LoRA + linear-head setup (r5) beats every generative variant on every metric, eliminates parse failures, and produces calibrated probabilities (AUC becomes meaningful).
2. **Class imbalance dominates PMCD.** Moderate is the majority class but has the lowest recall; class weights help but do not fully solve it — focal loss or resampling are worth trying.
3. **Cross-domain transfer is weak.** PMED → PMCD drops ~5 F1 relative to in-domain training; the lab-vs-clinical distribution shift is substantial.
4. **Phi-4-mini collapses on PMCD.** Despite matching Qwen3-4B on PMED AUC (0.94), Phi-4-mini's PMCD clf_head collapses to the majority class. Qwen3-4B is the most stable choice in the 4B tier.
5. **RL helps recall but not Macro F1.** GRPO / SRPO restore No-pain and Severe recall that SFT lost, but never outperform the classification head on aggregate metrics.

## TODO

- [x] Classical RF / SVM baselines (LOSO-CV)
- [x] Attention-fusion model on PMED and PMCD (LOSO-CV, GPU)
- [x] CrossMod-Transformer (FCN + ALSTM + cross-attention fusion) on PMED / PMCD
- [x] Qwen3-4B SFT baseline (2-stage, CoT output)
- [x] GRPO / SRPO reinforcement learning
- [x] Classification-head fine-tuning (**current best**)
- [x] Phi-4-mini comparison (SFT + clf_head)
- [x] PMED → PMCD transfer evaluation
- [ ] Transfer learning experiments (scratch vs frozen vs finetune)
- [ ] Modality ablation (BVP / EDA / EMG / Resp individually)
- [ ] Focal loss / resampling on PMCD Moderate
- [ ] DeepSeek-R1-Distill-Qwen-7B clf_head comparison
- [ ] Attention-weight visualization across PMED vs PMCD
- [ ] Temperature confounding ablation on PMED
- [ ] Update paper with full experimental results

## Label Schemes

| Scheme | PMED (heater) | PMCD |
|---|---|---|
| `binary` | no-pain / pain | no-pain / pain |
| `3class` | no-pain / lower-pain / higher-pain | no-pain / moderate / severe |
| `full` | 6 classes (baseline, NP, P1–P4) | 3 classes (native) |
