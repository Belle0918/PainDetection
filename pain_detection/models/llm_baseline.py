"""
LLM-based baseline for pain detection.

Serialises extracted physiological features into a structured text prompt
and queries Claude to classify pain level (zero-shot or few-shot).

This is an exploratory baseline to test whether LLM world knowledge
provides useful signal for physiological pain classification.

Two prompt modes:
  - "raw"      : original numeric feature dump (all 64 features)
  - "semantic" : natural-language physiological summary with percentile
                 levels computed from the training population
"""
from __future__ import annotations
import numpy as np
import time
import random
from typing import Optional


# ── Key features per modality (used for semantic summaries) ──────────────────

_MODALITY_KEYS = {
    "Bvp": {
        "full_name": "Blood Volume Pulse (BVP) — reflects cardiac activity",
        "features": {
            "std":          ("amplitude variability",   "higher during sympathetic arousal"),
            "range":        ("pulse amplitude range",   "wider range may indicate stress response"),
            "kurtosis":     ("waveform peakedness",     "changes with vascular tone"),
            "spec_entropy": ("spectral regularity",     "lower = more regular heart rhythm"),
        },
    },
    "Eda_E4": {
        "full_name": "Electrodermal Activity (EDA) — skin conductance, reflects sympathetic nervous system",
        "features": {
            "mean":         ("tonic skin conductance level", "higher = greater sympathetic arousal"),
            "std":          ("phasic EDA fluctuation",       "higher = more frequent skin conductance responses"),
            "range":        ("EDA response amplitude",       "larger = stronger autonomic reactions"),
            "spec_entropy": ("EDA spectral complexity",      "higher = more variable autonomic drive"),
        },
    },
    "Emg": {
        "full_name": "Electromyography (EMG) — muscle electrical activity",
        "features": {
            "std":          ("muscle activation level",  "higher = greater muscle tension or guarding"),
            "rms":          ("RMS muscle energy",        "higher during pain-related muscle contraction"),
            "zcr":          ("signal zero-crossing rate", "reflects muscle firing frequency"),
            "spec_entropy": ("EMG spectral complexity",   "higher = more distributed muscle firing"),
        },
    },
    "Resp": {
        "full_name": "Respiration — breathing pattern",
        "features": {
            "std":          ("breathing depth variability", "pain may alter breathing depth"),
            "range":        ("breath amplitude range",      "restricted or exaggerated during pain"),
            "skew":         ("inhalation/exhalation asymmetry", "changes with pain guarding"),
            "spec_entropy": ("breathing regularity",        "lower = more regular breathing"),
        },
    },
}


def _format_features_raw(features: dict[str, float], context: str | None = None) -> str:
    """Convert a feature dict into a readable text block (original mode)."""
    lines = []
    if context:
        lines.append(f"Recording context: {context}")
    lines.append("Physiological features:")
    for key, val in features.items():
        lines.append(f"  {key}: {val:.4f}")
    return "\n".join(lines)


def _percentile_label(value: float, pcts: dict) -> str:
    """Return a qualitative level based on population percentiles."""
    if value <= pcts["p10"]:
        return "very low"
    elif value <= pcts["p25"]:
        return "low"
    elif value <= pcts["p75"]:
        return "normal"
    elif value <= pcts["p90"]:
        return "high"
    else:
        return "very high"


def _format_features_semantic(
    features: dict[str, float],
    pop_stats: dict[str, dict],
    context: str | None = None,
) -> str:
    """Convert features into a concise physiological narrative."""
    lines = []
    if context:
        lines.append(f"Recording context: {context}")
    lines.append("")

    for modality, info in _MODALITY_KEYS.items():
        lines.append(f"[{info['full_name']}]")
        for feat_suffix, (description, interpretation) in info["features"].items():
            key = f"{modality}_{feat_suffix}"
            if key not in features:
                continue
            val = features[key]
            stats_key = key
            if stats_key in pop_stats:
                level = _percentile_label(val, pop_stats[stats_key])
            else:
                level = "unknown"
            lines.append(f"  {description}: {level} ({val:.4f}) — {interpretation}")
        lines.append("")

    return "\n".join(lines)


def _build_system_prompt_raw(class_names: list[str], context: str | None = None) -> str:
    """Build the system prompt for raw mode."""
    classes_str = ", ".join(f'"{name}"' for name in class_names)
    prompt = (
        "You are a physiological signal analysis expert. "
        "Your task is to classify pain levels from extracted physiological features.\n\n"
        "The features come from biosensors measuring blood volume pulse (BVP), "
        "electrodermal activity (EDA), electromyography (EMG), and respiration (Resp). "
        "Each sensor has statistical features (mean, std, range, skew, kurtosis), "
        "energy features (RMS, MAD, zero-crossing rate), and frequency features "
        "(dominant frequency, spectral entropy, low/high band power).\n\n"
    )
    if context:
        prompt += f"The recording was made in a {context} setting.\n\n"
    prompt += (
        f"Classify the sample into exactly one of these classes: {classes_str}.\n"
        "Respond with ONLY the class name, nothing else."
    )
    return prompt


def _build_system_prompt_semantic(class_names: list[str], context: str | None = None) -> str:
    """Build the system prompt for semantic mode with domain knowledge."""
    classes_str = ", ".join(f'"{name}"' for name in class_names)
    prompt = (
        "You are a physiological signal analysis expert specializing in pain assessment.\n\n"
        "You will receive a summary of physiological signals from 4 biosensors. "
        "Each measurement is described with its qualitative level (very low / low / "
        "normal / high / very high) relative to the study population, the raw value, "
        "and a brief physiological interpretation.\n\n"
        "Key pain indicators to consider:\n"
        "- EDA: Pain activates the sympathetic nervous system, typically INCREASING "
        "skin conductance level and phasic responses.\n"
        "- EMG: Pain often causes involuntary muscle guarding, INCREASING muscle tension.\n"
        "- BVP: Sympathetic arousal can alter heart rate variability and pulse amplitude.\n"
        "- Respiration: Pain may cause breath-holding, shallow breathing, or irregular patterns.\n\n"
        "IMPORTANT: A single elevated indicator is not sufficient. Look for CONVERGING "
        "evidence across multiple modalities. Pain typically shows a pattern of elevated "
        "EDA + increased EMG + altered BVP/Resp, not just one isolated change.\n\n"
    )
    if context:
        prompt += f"The recording was made in a {context} setting.\n\n"
    prompt += (
        f"Classify the sample into exactly one of these classes: {classes_str}.\n"
        "Respond with ONLY the class name, nothing else."
    )
    return prompt


def _build_fewshot_examples(
    feat_texts: list[str],
    labels: list[str],
) -> list[dict[str, str]]:
    """Build few-shot example messages (user/assistant pairs)."""
    messages = []
    for text, label in zip(feat_texts, labels):
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": label})
    return messages


class LLMBaseline:
    """LLM-based pain classifier using Claude API.

    Parameters
    ----------
    class_names    : list of class name strings, ordered by class index
    context        : optional context string ('experimental' or 'clinical')
    n_shots        : number of few-shot examples per class (0 = zero-shot)
    model          : Claude model ID
    max_retries    : retry count for API errors
    random_state   : seed for few-shot example selection
    prompt_mode    : "raw" (original 64-feature dump) or "semantic" (natural language)
    """

    def __init__(
        self,
        class_names: list[str],
        context: str | None = None,
        n_shots: int = 0,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
        random_state: int = 42,
        prompt_mode: str = "semantic",
    ):
        import anthropic
        self.client = anthropic.Anthropic()
        self.class_names = class_names
        self.context = context
        self.n_shots = n_shots
        self.model = model
        self.max_retries = max_retries
        self.rng = random.Random(random_state)
        self.prompt_mode = prompt_mode

        self._fewshot_messages: list[dict] = []
        self._pop_stats: dict[str, dict] = {}

        # For storing training data (used to pick few-shot examples)
        self._train_feat_dicts: list[dict] | None = None
        self._train_labels: np.ndarray | None = None

    def _compute_pop_stats(self, X_features: np.ndarray, feature_names: list[str]):
        """Compute population percentiles from training data for semantic mode."""
        self._pop_stats = {}
        for j, name in enumerate(feature_names):
            vals = X_features[:, j]
            self._pop_stats[name] = {
                "p10": float(np.percentile(vals, 10)),
                "p25": float(np.percentile(vals, 25)),
                "p50": float(np.percentile(vals, 50)),
                "p75": float(np.percentile(vals, 75)),
                "p90": float(np.percentile(vals, 90)),
            }

    def _format_one(self, feat_dict: dict[str, float]) -> str:
        """Format a single sample according to prompt_mode."""
        if self.prompt_mode == "semantic" and self._pop_stats:
            return _format_features_semantic(feat_dict, self._pop_stats, self.context)
        else:
            return _format_features_raw(feat_dict, self.context)

    def fit(self, X_features: np.ndarray, y: np.ndarray,
            feature_names: list[str] | None = None):
        """Store training data for few-shot example selection.

        Parameters
        ----------
        X_features   : (N, F) feature matrix (from extract_features)
        y            : (N,) integer labels
        feature_names: column names for the features
        """
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X_features.shape[1])]

        self._feature_names = feature_names

        # Compute population stats for semantic mode
        if self.prompt_mode == "semantic":
            self._compute_pop_stats(X_features, feature_names)

        # Build system prompt (needs to happen after pop_stats for semantic)
        if self.prompt_mode == "semantic":
            self._system_prompt = _build_system_prompt_semantic(self.class_names, self.context)
        else:
            self._system_prompt = _build_system_prompt_raw(self.class_names, self.context)

        self._train_feat_dicts = []
        for i in range(len(X_features)):
            d = {feature_names[j]: float(X_features[i, j])
                 for j in range(len(feature_names))}
            self._train_feat_dicts.append(d)
        self._train_labels = y.copy()

        # Pre-select few-shot examples if needed
        if self.n_shots > 0:
            self._fewshot_messages = self._select_fewshot_examples()

        return self

    def _select_fewshot_examples(self) -> list[dict]:
        """Select n_shots examples per class from training data."""
        examples_text = []
        examples_label = []
        for cls_idx, cls_name in enumerate(self.class_names):
            indices = np.where(self._train_labels == cls_idx)[0].tolist()
            if not indices:
                continue
            chosen = self.rng.sample(indices, min(self.n_shots, len(indices)))
            for idx in chosen:
                text = self._format_one(self._train_feat_dicts[idx])
                examples_text.append(text)
                examples_label.append(cls_name)
        # Shuffle so classes are interleaved
        combined = list(zip(examples_text, examples_label))
        self.rng.shuffle(combined)
        texts, labels = zip(*combined) if combined else ([], [])
        return _build_fewshot_examples(list(texts), list(labels))

    def _classify_one(self, feat_dict: dict[str, float]) -> int:
        """Classify a single sample via the LLM API."""
        user_msg = self._format_one(feat_dict)

        messages = list(self._fewshot_messages)  # few-shot examples
        messages.append({"role": "user", "content": user_msg})

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=20,
                    system=self._system_prompt,
                    messages=messages,
                )
                answer = response.content[0].text.strip()
                return self._parse_answer(answer)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  API error after {self.max_retries} retries: {e}")
                    return 0  # fallback to class 0

    def _parse_answer(self, answer: str) -> int:
        """Map LLM text response to class index."""
        answer_lower = answer.lower().strip().strip('"').strip("'")
        for idx, name in enumerate(self.class_names):
            if name.lower() in answer_lower:
                return idx
        # Try partial match
        for idx, name in enumerate(self.class_names):
            for word in name.lower().split():
                if word in answer_lower and word not in ("no", "pain", "a", "the"):
                    return idx
        # If "no" appears, likely no-pain
        if "no" in answer_lower:
            return 0
        # Default: highest class (pain)
        return len(self.class_names) - 1

    def predict(self, X_features: np.ndarray) -> np.ndarray:
        """Classify all samples. X_features: (N, F) feature matrix."""
        feature_names = getattr(self, '_feature_names', None)
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X_features.shape[1])]

        preds = []
        for i in range(len(X_features)):
            feat_dict = {feature_names[j]: float(X_features[i, j])
                         for j in range(len(feature_names))}
            pred = self._classify_one(feat_dict)
            preds.append(pred)
            if (i + 1) % 10 == 0:
                print(f"    Classified {i+1}/{len(X_features)}")
        return np.array(preds)

    def predict_proba(self, X_features: np.ndarray) -> np.ndarray:
        """LLM gives hard predictions; return one-hot probabilities."""
        preds = self.predict(X_features)
        proba = np.zeros((len(preds), len(self.class_names)))
        proba[np.arange(len(preds)), preds] = 1.0
        return proba
