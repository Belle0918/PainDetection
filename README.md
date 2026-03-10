# Automated Pain Detection from Physiological Signals

Exploring automated pain level detection by combining multiple physiological signals, with cross-domain generalization from experimental to clinical settings.

## Dataset

This project uses the **PainMonit Dataset (PMD)** collected by Gouverneur et al. (2024), which consists of two parts:

- **PMED**: Heat-induced pain in 52 healthy subjects (6 signals: BVP, EDA×2, ECG, EMG, Respiration)
- **PMCD**: Physiotherapy fascial therapy pain in 49 patients (7 signals: BVP×2, EDA×2, EMG, Temperature, Respiration)

> If you use the PMD dataset, please cite the original paper:
> Gouverneur et al., "An Experimental and Clinical Physiological Signal Dataset for Automated Pain Recognition", *Scientific Data*, 2024. https://doi.org/10.1038/s41597-024-03878-w

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

Place raw datasets in `PMED/dataset/raw-data/` and `PMCD/dataset/raw-data/`, then generate numpy files:

```bash
cd PMED && python create_np_files.py && cd ..
cd PMCD && python create_np_files.py && cd ..
```

## Experiments

### Baseline (Random Forest, LOSO-CV)

```bash
# PMED, subjective CoVAS label, 3-class severity
python run_baseline.py --dataset pmed --label covas --scheme 3class --model rf

# PMCD, 3-class (no-pain / moderate / severe)
python run_baseline.py --dataset pmcd --scheme 3class --model rf

# Use specific sensors only (minimum 3)
python run_baseline.py --dataset pmed --scheme 3class --sensors Bvp Eda_E4 Emg
```

### Cross-domain: PMED → PMCD

```bash
python run_cross_domain.py --model rf --scheme 3class --also-within
```

## Results

| Experiment | Accuracy | Macro F1 | AUC |
|---|---|---|---|
| PMED CoVAS 3-class (LOSO-CV) | 0.563 | 0.555 | 0.730 |
| PMED Heater 3-class (LOSO-CV) | 0.509 | 0.506 | 0.705 |
| PMCD 3-class (LOSO-CV) | 0.576 | 0.483 | 0.688 |
| Cross-domain PMED → PMCD | 0.388 | 0.366 | 0.552 |

Chance level for 3-class: 33.3%. Cross-domain accuracy near chance indicates that models trained on laboratory heat pain do not generalize to real clinical pain.

## Future Work

We plan to benchmark LLM-based classifiers directly on the extracted physiological feature tables to evaluate whether LLMs with physiological domain knowledge can match or explain traditional ML baselines.

## Label Schemes

| Scheme | PMED (heater) | PMCD |
|---|---|---|
| `binary` | no-pain / pain | no-pain / pain |
| `3class` | no-pain / lower-pain (P1-P2) / higher-pain (P3-P4) | no-pain / moderate / severe |
| `full` | 6 classes (baseline, NP, P1–P4) | 3 classes (native) |

PMCD labels use per-subject NRS thresholds from the original dataset. PMED CoVAS `3class` (subjective quartiles) is recommended over heater `3class` for pain severity analysis.
