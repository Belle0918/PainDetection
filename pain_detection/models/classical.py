"""
Feature-based baselines: Random Forest and SVM with standardisation.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_rf(n_estimators: int = 200, random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )),
    ])


def build_svm(C: float = 1.0, random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(
            C=C,
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        )),
    ])
