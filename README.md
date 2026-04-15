# Cross-Context Physiological Pain Recognition

Automated pain recognition from multimodal physiological signals, studying cross-context transfer from experimental (PMED) to clinical (PMCD) settings on the PainMonit Database.

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
│       ├── transfer.py            # PMED→PMCD transfer learning wrapper
│       └── llm_baseline.py        # LLM-based classification (Claude API)
│
├── experiments/                   # Experiment scripts
│   ├── run_baseline.py            # RF / SVM / CNN baselines with LOSO-CV
│   ├── run_cross_domain.py        # Cross-domain: train PMED, test PMCD
│   ├── run_attention_fusion.py    # Attention-fusion model with LOSO-CV
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
| **Transfer Learning** | Pretrain on PMED → fine-tune on PMCD (frozen / full) | `run_transfer.py` |
| LLM Baseline | Claude API zero-shot / few-shot on extracted features | `run_llm_baseline.py` |

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

# Transfer learning (scratch vs frozen vs finetune)
python experiments/run_transfer.py --scheme 3class

# Modality ablation (single sensor)
python experiments/run_attention_fusion.py --dataset pmed --scheme binary --sensors Bvp
python experiments/run_attention_fusion.py --dataset pmed --scheme binary --sensors Eda_E4

# LLM baseline (requires ANTHROPIC_API_KEY)
python experiments/run_llm_baseline.py --dataset pmcd --scheme 3class --compare --context
```

## Preliminary Results (Classical Baselines)

| Setting | Model | Accuracy | Macro F1 | AUC |
|---|---|---|---|---|
| PMED CoVAS 3-class | Random Forest | 0.563 | 0.555 | 0.730 |
| PMED Heater 3-class | Random Forest | 0.509 | 0.506 | 0.705 |
| PMCD 3-class | Random Forest | 0.576 | 0.483 | 0.688 |
| Cross-domain PMED→PMCD | Random Forest | 0.388 | 0.366 | 0.552 |

Chance level for 3-class: 33.3%. Cross-domain accuracy near chance confirms a large domain gap between experimental and clinical pain.

## TODO

- [ ] Run attention-fusion model on PMED and PMCD (LOSO-CV, GPU)
- [ ] Run transfer learning experiments (scratch vs frozen vs finetune)
- [ ] Run modality ablation (BVP / EDA / EMG / Resp individually)
- [ ] Run LLM baseline (zero-shot and few-shot)
- [ ] Temperature ablation (include skin temp, check confounding on PMED)
- [ ] Attention weight visualization across PMED vs PMCD
- [ ] Update paper with full experimental results

## Label Schemes

| Scheme | PMED (heater) | PMCD |
|---|---|---|
| `binary` | no-pain / pain | no-pain / pain |
| `3class` | no-pain / lower-pain / higher-pain | no-pain / moderate / severe |
| `full` | 6 classes (baseline, NP, P1–P4) | 3 classes (native) |
