import json
from typing import Dict

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from agent.config import METRICS_PATH


LABEL_ORDER = ["cancerous", "non_cancerous"]


def evaluate_model(model, X_test, y_test) -> Dict:
    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_matrix_labels": LABEL_ORDER,
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=LABEL_ORDER,
            zero_division=0,
            output_dict=True,
        ),
    }


def save_metrics(metrics: Dict) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def load_metrics() -> Dict:
    if not METRICS_PATH.exists():
        return {}
    with METRICS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)
