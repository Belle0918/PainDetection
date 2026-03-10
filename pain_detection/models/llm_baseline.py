"""
LLM-based baseline for pain detection.

Serialises extracted physiological features into a structured text prompt
and queries Claude to classify pain level (zero-shot or few-shot).

This is an exploratory baseline to test whether LLM world knowledge
provides useful signal for physiological pain classification.
"""
from __future__ import annotations
import numpy as np
import time
import random
from typing import Optional


def _format_features(features: dict[str, float], context: str | None = None) -> str:
    """Convert a feature dict into a readable text block."""
    lines = []
    if context:
        lines.append(f"Recording context: {context}")
    lines.append("Physiological features:")
    for key, val in features.items():
        lines.append(f"  {key}: {val:.4f}")
    return "\n".join(lines)


def _build_system_prompt(class_names: list[str], context: str | None = None) -> str:
    """Build the system prompt explaining the classification task."""
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


def _build_fewshot_examples(
    feat_dicts: list[dict[str, float]],
    labels: list[str],
    context: str | None = None,
) -> list[dict[str, str]]:
    """Build few-shot example messages (user/assistant pairs)."""
    messages = []
    for feat, label in zip(feat_dicts, labels):
        messages.append({
            "role": "user",
            "content": _format_features(feat, context),
        })
        messages.append({
            "role": "assistant",
            "content": label,
        })
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
    """

    def __init__(
        self,
        class_names: list[str],
        context: str | None = None,
        n_shots: int = 0,
        model: str = "claude-sonnet-4-5",
        max_retries: int = 3,
        random_state: int = 42,
    ):
        import anthropic
        self.client = anthropic.Anthropic()
        self.class_names = class_names
        self.context = context
        self.n_shots = n_shots
        self.model = model
        self.max_retries = max_retries
        self.rng = random.Random(random_state)

        self._system_prompt = _build_system_prompt(class_names, context)
        self._fewshot_messages: list[dict] = []

        # For storing training data (used to pick few-shot examples)
        self._train_feat_dicts: list[dict] | None = None
        self._train_labels: np.ndarray | None = None

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

        self._train_feat_dicts = []
        for i in range(len(X_features)):
            d = {feature_names[j]: float(X_features[i, j])
                 for j in range(len(feature_names))}
            self._train_feat_dicts.append(d)
        self._train_labels = y.copy()
        self._feature_names = feature_names

        # Pre-select few-shot examples if needed
        if self.n_shots > 0:
            self._fewshot_messages = self._select_fewshot_examples()

        return self

    def _select_fewshot_examples(self) -> list[dict]:
        """Select n_shots examples per class from training data."""
        examples_feat = []
        examples_label = []
        for cls_idx, cls_name in enumerate(self.class_names):
            indices = np.where(self._train_labels == cls_idx)[0].tolist()
            if not indices:
                continue
            chosen = self.rng.sample(indices, min(self.n_shots, len(indices)))
            for idx in chosen:
                examples_feat.append(self._train_feat_dicts[idx])
                examples_label.append(cls_name)
        # Shuffle so classes are interleaved
        combined = list(zip(examples_feat, examples_label))
        self.rng.shuffle(combined)
        feats, labels = zip(*combined) if combined else ([], [])
        return _build_fewshot_examples(list(feats), list(labels), self.context)

    def _classify_one(self, feat_dict: dict[str, float]) -> int:
        """Classify a single sample via the LLM API."""
        user_msg = _format_features(feat_dict, self.context)

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

    # predict_proba is intentionally not implemented: LLM returns hard labels only.
    # evaluate.py will skip AUC when predict_proba is absent.
