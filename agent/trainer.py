import json
import logging
import warnings
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from agent.config import (
    BEST_MODEL_PATH,
    METRICS_PATH,
    SAMPLE_DATA_PATH,
    TRAINING_DATA_PATH,
    ensure_directories,
)
from agent.evaluator import evaluate_model, save_metrics
from agent.feature_extractor import extract_feature_dataframe


logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str], None]


def _notify(callback: Optional[ProgressCallback], step: str, message: str) -> None:
    logger.info("%s: %s", step, message)
    if callback:
        callback(step, message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _generate_synthetic_demo_data() -> pd.DataFrame:
    cancer_patterns = [
        "MKRKDEKRASMYCDEKRKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYKRDEKRASMYC",
        "MDEKRHDEKRHMYCAVILMFWYPPQRSTDEKRHDEKRHACDEFGHIKLMNPQRSTVWY",
        "MKLMNPQRSTVWYDEKRDEKRASMYCFGHIKLMNPQRSTVWYDEKRHDEKRHSTNQ",
        "MKRASMYCDEKRHSTNQACDEFGHIKLMNPQRSTVWYDEKRDEKRHMYCAVILMF",
        "MDEKRASDEKRHMYCKRHACDEFGHIKLMNPQRSTVWYKRHDEKRASMYCSTNQ",
    ]
    normal_patterns = [
        "MAVILMFWYPSTNQSTNQACDEFGHIKLMNPQRSTVWYAVILMFWYPSTNQSTNQ",
        "MGGAVILSTNQAVILMFWYPACDEFGHIKLMNPQRSTVWYGGSTNQAVILMFWY",
        "MSTNQAVILMFWYPGGACDEFGHIKLMNPQRSTVWYSTNQAVILMFWYPGG",
        "MAVILMFWYPGGSTNQACDEFGHIKLMNPQRSTVWYAVILMFWYPGGSTNQ",
        "MGGPPSTNQAVILMFWYACDEFGHIKLMNPQRSTVWYGGPPSTNQAVILMFWY",
    ]
    rows = []
    for index in range(30):
        base = cancer_patterns[index % len(cancer_patterns)]
        rotated = base[index % len(base) :] + base[: index % len(base)]
        rows.append({"sequence": rotated, "label": "cancerous", "source": "synthetic_demo"})
    for index in range(30):
        base = normal_patterns[index % len(normal_patterns)]
        rotated = base[index % len(base) :] + base[: index % len(base)]
        rows.append({"sequence": rotated, "label": "non_cancerous", "source": "synthetic_demo"})
    return pd.DataFrame(rows)


def _read_csv_if_valid(path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if {"sequence", "label"}.issubset(df.columns) and not df.empty:
            return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read %s: %s", path, exc)
    return None


def _has_two_classes(df: pd.DataFrame) -> bool:
    if df.empty or "label" not in df.columns:
        return False
    counts = df["label"].value_counts()
    return len(counts) >= 2 and counts.min() >= 2


def load_training_data() -> pd.DataFrame:
    processed = _read_csv_if_valid(TRAINING_DATA_PATH)
    if processed is not None and _has_two_classes(processed):
        return processed

    sample = _read_csv_if_valid(SAMPLE_DATA_PATH)
    if sample is not None and _has_two_classes(sample):
        sample = sample.copy()
        if "source" not in sample.columns:
            sample["source"] = "synthetic_demo"
        return sample

    return _generate_synthetic_demo_data()


def train_models(df: pd.DataFrame) -> Dict[str, Dict]:
    df = df.dropna(subset=["sequence", "label"]).copy()
    df["sequence"] = df["sequence"].astype(str)
    df["label"] = df["label"].astype(str)

    if not _has_two_classes(df):
        raise ValueError("Training requires at least two labels with at least two samples each.")

    X = extract_feature_dataframe(df["sequence"])
    y = df["label"]
    test_size = 0.25 if len(df) >= 20 else 0.4
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    models = {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
            ]
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        ),
        "SVC": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", SVC(probability=True, class_weight="balanced", random_state=42)),
            ]
        ),
    }

    results: Dict[str, Dict] = {}
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
        results[name] = {
            "model": model,
            "metrics": metrics,
            "feature_columns": list(X.columns),
        }
    return results


def select_best_model(results: Dict[str, Dict]) -> Tuple[str, object, Dict]:
    best_name = max(results, key=lambda name: results[name]["metrics"]["f1_score"])
    best = results[best_name]
    return best_name, best["model"], best["metrics"]


def _source_summary(df: pd.DataFrame) -> Dict:
    if "source" not in df.columns:
        return {"synthetic_demo": int(len(df))}
    return {str(key): int(value) for key, value in df["source"].fillna("unknown").value_counts().items()}


def save_best_model(model, metrics: Dict) -> None:
    ensure_directories()
    feature_columns = metrics.get("feature_columns", [])
    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "trained_at": metrics.get("last_trained_timestamp", _utc_now()),
        "label_order": ["cancerous", "non_cancerous"],
    }
    joblib.dump(artifact, BEST_MODEL_PATH)
    save_metrics(metrics)


def train_pipeline(use_public_search: bool = True, progress_callback: Optional[ProgressCallback] = None) -> Dict:
    ensure_directories()
    warnings: List[str] = []
    search_results = []
    download_log = []
    public_df = pd.DataFrame()

    _notify(progress_callback, "Loading environment variables", "Environment loaded from .env when present.")

    if use_public_search:
        from agent.data_cleaner import clean_and_label_records, save_training_data
        from agent.dataset_downloader import download_all
        from agent.dataset_search_agent import run_dataset_search, run_targeted_public_sequence_search

        _notify(progress_callback, "Searching datasets", "Searching targeted public UniProt and optional NCBI protein records.")
        search_results = run_targeted_public_sequence_search()
        if not search_results:
            _notify(progress_callback, "Searching datasets", "Targeted search found no records; trying broad public metadata search.")
            search_results = run_dataset_search()
        if not search_results:
            warnings.append("No public dataset metadata was found or internet access failed. Falling back to demo data.")

        _notify(progress_callback, "Downloading sequences", f"Attempting downloads from {len(search_results)} metadata results.")
        records, download_log = download_all(search_results)

        _notify(progress_callback, "Cleaning sequences", f"Cleaning and labeling {len(records)} downloaded record payloads.")
        if records:
            public_df = clean_and_label_records(records)
            if not public_df.empty:
                save_training_data(public_df)
        if public_df.empty:
            warnings.append("No usable labeled public sequences were collected. Using sample_data.csv or generated demo data.")
    else:
        warnings.append("Public search was skipped for this run; using available local data.")

    _notify(progress_callback, "Labeling data", "Loading the best available labeled training data.")
    df = load_training_data()
    if not _has_two_classes(df):
        warnings.append("Local data did not contain enough classes. Generated synthetic demo data was used.")
        df = _generate_synthetic_demo_data()

    if "source" not in df.columns:
        df["source"] = "unknown"

    _notify(progress_callback, "Extracting features", f"Extracting sequence features for {len(df)} samples.")
    _notify(progress_callback, "Training models", "Training Logistic Regression, Random Forest, and SVC models.")
    results = train_models(df)

    _notify(progress_callback, "Evaluating models", "Selecting the model with the best weighted F1-score.")
    best_model_name, best_model, best_metrics = select_best_model(results)

    label_counts = df["label"].value_counts().to_dict()
    trained_from = "public_data" if use_public_search and not public_df.empty and TRAINING_DATA_PATH.exists() else "synthetic_fallback"
    all_model_metrics = {
        name: payload["metrics"] for name, payload in results.items()
    }
    final_metrics = {
        **best_metrics,
        "best_model_name": best_model_name,
        "saved_model_path": str(BEST_MODEL_PATH),
        "metrics_path": str(METRICS_PATH),
        "total_samples": int(len(df)),
        "cancerous_samples": int(label_counts.get("cancerous", 0)),
        "non_cancerous_samples": int(label_counts.get("non_cancerous", 0)),
        "source_summary": _source_summary(df),
        "dataset_metadata_results": int(len(search_results)),
        "download_attempts": int(len(download_log)),
        "trained_from": trained_from,
        "last_trained_timestamp": _utc_now(),
        "warnings": warnings,
        "data_quality_note": (
            "Public-data training uses authentic sequences from public protein databases, but labels are inferred "
            "from source query terms and metadata. Replace with expert-curated biological labels before research use."
        ),
        "all_model_metrics": all_model_metrics,
        "feature_columns": results[best_model_name]["feature_columns"],
        "feature_count": int(len(results[best_model_name]["feature_columns"])),
    }

    _notify(progress_callback, "Saving best model", f"Saving {best_model_name} to {BEST_MODEL_PATH}.")
    save_best_model(best_model, final_metrics)
    return final_metrics


def metrics_as_json() -> str:
    if not METRICS_PATH.exists():
        return "{}"
    return json.dumps(json.loads(METRICS_PATH.read_text(encoding="utf-8")), indent=2)
