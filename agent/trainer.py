import json
import logging
import warnings
from collections import Counter
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
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
from agent.api_audit import load_api_audit, reset_api_audit, summarize_api_audit
from agent.evaluator import evaluate_model, save_metrics
from agent.feature_extractor import extract_feature_dataframe, kmer_counts


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
    test_size = 0.1 if len(df) >= 100 else 0.3
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
            n_estimators=220,
            random_state=42,
            class_weight="balanced",
            max_features=0.5,
            n_jobs=1,
        ),
        "ExtraTreesClassifier": ExtraTreesClassifier(
            n_estimators=260,
            random_state=42,
            class_weight="balanced",
            max_features=0.5,
            n_jobs=1,
        ),
        "SVC_RBF": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", SVC(probability=True, class_weight="balanced", random_state=42)),
            ]
        ),
        "SVC_Linear": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42)),
            ]
        ),
        "ComplementNB": ComplementNB(alpha=0.25),
        "KNeighborsClassifier": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ]
        ),
    }

    results: Dict[str, Dict] = {}
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
            metrics.update(
                {
                    "train_samples": int(len(X_train)),
                    "test_samples": int(len(X_test)),
                    "test_size": float(test_size),
                    "split_random_state": 42,
                }
            )
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


def _label_kmer_markers(df: pd.DataFrame, k: int = 3, limit: int = 10) -> List[Dict]:
    labels = ["cancerous", "non_cancerous"]
    counts_by_label = {label: Counter() for label in labels}
    totals_by_label = {label: 0 for label in labels}

    for label in labels:
        for sequence in df.loc[df["label"] == label, "sequence"].astype(str):
            counts = kmer_counts(sequence, k)
            counts_by_label[label].update(counts)
            totals_by_label[label] += sum(counts.values())

    rows: List[Dict] = []
    for label in labels:
        other_label = "non_cancerous" if label == "cancerous" else "cancerous"
        total = max(totals_by_label[label], 1)
        other_total = max(totals_by_label[other_label], 1)
        candidates = set(counts_by_label[label]) | set(counts_by_label[other_label])
        scored = []
        for kmer in candidates:
            frequency = counts_by_label[label][kmer] / total
            other_frequency = counts_by_label[other_label][kmer] / other_total
            score = frequency - other_frequency
            if score > 0:
                scored.append((score, kmer, frequency, other_frequency))
        for score, kmer, frequency, other_frequency in sorted(scored, reverse=True)[:limit]:
            rows.append(
                {
                    "label": label,
                    "k": k,
                    "kmer": kmer,
                    "enrichment_score": float(score),
                    "label_frequency": float(frequency),
                    "other_label_frequency": float(other_frequency),
                }
            )
    return rows


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
    merge_info = {
        "previous_samples": 0,
        "new_samples": 0,
        "new_unique_samples_added": 0,
        "merged_samples": 0,
        "duplicates_removed": 0,
        "conflicting_sequences_removed": 0,
    }

    _notify(progress_callback, "Loading environment variables", "Environment loaded from .env when present.")

    if use_public_search:
        reset_api_audit()
        from agent.data_cleaner import clean_and_label_records, merge_with_existing_training_data
        from agent.dataset_downloader import download_all
        from agent.dataset_search_agent import run_dataset_search, run_targeted_public_sequence_search

        _notify(
            progress_callback,
            "Searching datasets",
            (
                "Searching targeted public UniProt and optional NCBI protein records, rotating result pages, "
                "and skipping source IDs already present in the committed training set."
            ),
        )
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
                merge_info = merge_with_existing_training_data(public_df)
                _notify(
                    progress_callback,
                    "Cleaning sequences",
                    (
                        f"Merged {merge_info['new_samples']} newly collected samples with "
                        f"{merge_info['previous_samples']} existing samples; "
                        f"{merge_info.get('new_unique_samples_added', 0)} unique new samples were added and "
                        f"{merge_info['merged_samples']} cumulative samples are available."
                    ),
                )
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
    _notify(
        progress_callback,
        "Training models",
        (
            "Training Logistic Regression, Random Forest, Extra Trees, SVC, Complement Naive Bayes, "
            "and KNN models."
        ),
    )
    results = train_models(df)

    _notify(progress_callback, "Evaluating models", "Selecting the model with the best weighted F1-score.")
    best_model_name, best_model, best_metrics = select_best_model(results)

    label_counts = df["label"].value_counts().to_dict()
    trained_from = "public_data" if TRAINING_DATA_PATH.exists() and not df.empty else "synthetic_fallback"
    all_model_metrics = {
        name: payload["metrics"] for name, payload in results.items()
    }
    api_audit_rows = load_api_audit(limit=0) if use_public_search else []
    api_audit_summary = summarize_api_audit(api_audit_rows)
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
        **api_audit_summary,
        "recent_api_calls": api_audit_rows[-75:],
        "trained_from": trained_from,
        "last_trained_timestamp": _utc_now(),
        "cumulative_training_enabled": True,
        "previous_training_samples": int(merge_info.get("previous_samples", 0)),
        "new_training_samples_collected": int(merge_info.get("new_samples", 0)),
        "new_unique_training_samples_added": int(merge_info.get("new_unique_samples_added", 0)),
        "cumulative_training_samples": int(merge_info.get("merged_samples", len(df))),
        "duplicate_sequences_removed": int(merge_info.get("duplicates_removed", 0)),
        "conflicting_sequences_removed": int(merge_info.get("conflicting_sequences_removed", 0)),
        "training_data_persistence_note": (
            "Each Train click merges newly collected valid sequences with the existing processed training set "
            "before retraining. The committed data/processed/training_data.csv file is used as the free persistent "
            "baseline on every rebuild. On Render Free, new runtime training data can still reset unless the updated "
            "processed CSV is committed back to GitHub."
        ),
        "warnings": warnings,
        "data_quality_note": (
            "Public-data training uses authentic sequences from public protein databases, but labels are inferred "
            "from source query terms and metadata. Replace with expert-curated biological labels before research use."
        ),
        "accuracy_improvement_note": (
            "The model now uses amino acid composition, dipeptide composition, hashed 3/4-mer composition, k-mer pattern "
            "statistics, and physicochemical features. It compares Logistic Regression, Random Forest, Extra Trees, "
            "SVC, Complement Naive Bayes, and KNN, then selects the best model by weighted F1-score. The model uses "
            "Render-friendly hashed k-mer bins to keep memory below the free-tier limit. The headline "
            "metric uses a stratified 10% holdout when at least 100 samples are available, which is closer to the "
            "small-holdout style used in many portfolio sequence-classification projects."
        ),
        "api_audit_note": (
            "Public API calls are executed server-side by Streamlit, so browser DevTools may only show Streamlit "
            "websocket traffic. Use the Recent API Calls table in the app to inspect UniProt, NCBI, Zenodo, and Kaggle activity."
        ),
        "kmer_marker_summary": _label_kmer_markers(df, k=3, limit=10),
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
