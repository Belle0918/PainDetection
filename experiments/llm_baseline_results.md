# LLM Baseline Results

## Experiment Setup

- **Model**: `claude-sonnet-4-20250514`
- **Prompt mode**: Semantic (natural-language physiological summaries with population-relative percentile levels)
- **Evaluation**: 80/20 subject split (not LOSO, to limit API cost)
- **Context**: Included in prompt (`controlled experimental (heat pain)`)
- **Random seed**: 42

### Prompt Design (Semantic Mode)

Each sample is represented as a structured physiological summary with 16 key features (4 modalities x 4 features each):

| Modality | Features | Pain Relevance |
|----------|----------|----------------|
| BVP (Blood Volume Pulse) | amplitude variability, pulse range, waveform peakedness, spectral regularity | Sympathetic arousal alters heart rate variability and pulse amplitude |
| EDA (Electrodermal Activity) | tonic level, phasic fluctuation, response amplitude, spectral complexity | Pain activates sympathetic NS, increasing skin conductance |
| EMG (Electromyography) | activation level, RMS energy, zero-crossing rate, spectral complexity | Pain causes involuntary muscle guarding |
| Respiration | depth variability, amplitude range, inhalation/exhalation asymmetry, regularity | Pain may cause breath-holding or irregular patterns |

Each feature is described with:
- A qualitative level (`very low` / `low` / `normal` / `high` / `very high`) relative to the training population (percentile-based)
- The raw numeric value
- A brief physiological interpretation

The system prompt includes domain knowledge about pain physiology and instructs the model to look for converging evidence across multiple modalities.

## PMED Binary Classification (No-pain vs Pain)

| Config | Accuracy | Macro F1 |
|--------|----------|----------|
| Zero-shot | 0.5400 | 0.5398 |
| Few-shot (k=3) | **0.5600** | **0.5593** |
| Few-shot (k=5) | 0.5400 | 0.5393 |

### Per-class Breakdown (Best: Few-shot k=3)

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| No-pain | 0.56 | 0.60 | 0.58 | 50 |
| Pain | 0.57 | 0.52 | 0.54 | 50 |

### Comparison: Raw vs Semantic Prompt Mode

| Config | Raw Accuracy | Raw F1 | Semantic Accuracy | Semantic F1 |
|--------|-------------|--------|-------------------|-------------|
| Zero-shot | 0.4800 | 0.3200 | **0.5400** | **0.5398** |
| Few-shot (k=3) | 0.4800 | 0.3400 | **0.5600** | **0.5593** |
| Few-shot (k=5) | — | — | 0.5400 | 0.5393 |

**Key improvement**: Semantic mode eliminates the class prediction bias seen in raw mode (raw zero-shot predicted 96% as Pain; raw few-shot k=3 predicted 94% as No-pain). Semantic mode achieves balanced predictions across both classes.

## PMCD 3-Class Classification (No-pain vs Moderate vs Severe)

| Config | Accuracy | Macro F1 |
|--------|----------|----------|
| Zero-shot | 0.4111 | 0.3950 |
| Few-shot (k=3) | **0.4333** | **0.4309** |
| Few-shot (k=5) | 0.4111 | 0.3975 |

### Per-class Breakdown (Best: Few-shot k=3)

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| No-pain | 0.38 | 0.40 | 0.39 | 30 |
| Moderate | 0.38 | 0.33 | 0.36 | 30 |
| Severe | 0.53 | 0.57 | 0.55 | 30 |

### Per-class Breakdown (Zero-shot)

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| No-pain | 0.39 | 0.57 | 0.46 | 30 |
| Moderate | 0.38 | 0.47 | 0.42 | 30 |
| Severe | 0.67 | 0.20 | 0.31 | 30 |

**Observation**: Zero-shot has high precision for Severe (0.67) but very low recall (0.20) — it is conservative about predicting severe pain. Few-shot k=3 improves Severe recall to 0.57 at the cost of slightly lower No-pain/Moderate performance, leading to better overall balance.

## Analysis

- **PMED binary**: LLM performance is near chance (54-56%), consistent with the small effect sizes between pain and no-pain physiological features (largest Cohen's d ~ 0.27 for EDA std)
- **PMCD 3-class**: Above the 33.3% chance level at 43%, with the best discrimination on the Severe class — likely because severe pain produces more distinct physiological patterns
- **Semantic prompting helps**: +6 points in PMED accuracy and +22 points in Macro F1 over raw feature dumps, primarily by eliminating single-class prediction bias
- **Few-shot provides marginal gains**: k=3 consistently outperforms zero-shot on both datasets; k=5 does not improve further
- **Input token reduction**: Semantic mode uses ~1,500 tokens per sample vs ~6,900 for raw mode (78% reduction)
- **Cross-dataset pattern**: PMCD is harder than PMED (weaker labels, more classes), but the LLM shows a useful signal for severe pain detection even in the clinical setting
- **Limitations**: Not LOSO evaluation; small test sets (100 PMED / 90 PMCD samples); LLM operates on extracted features, not raw waveforms
