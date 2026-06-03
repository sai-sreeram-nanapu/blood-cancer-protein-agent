import logging
from typing import Dict

import joblib

from agent.config import BEST_MODEL_PATH
from agent.data_cleaner import clean_sequence, is_valid_sequence, parse_fasta_records
from agent.evaluator import load_metrics
from agent.feature_extractor import (
    amino_acid_composition,
    extract_feature_dataframe,
    extract_features,
    top_kmers,
)
from agent.trainer import train_pipeline


logger = logging.getLogger(__name__)


def load_model():
    if not BEST_MODEL_PATH.exists():
        logger.info("No saved model found. Training fallback model from local data.")
        train_pipeline(use_public_search=False)
    return joblib.load(BEST_MODEL_PATH)


def _probabilities(model, X) -> Dict[str, float]:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        return {label: float(prob) for label, prob in zip(model.classes_, probabilities)}
    return {}


def _top_features(features: Dict[str, float], limit: int = 12) -> Dict[str, float]:
    excluded = {"sequence_length", "approx_molecular_weight"}
    candidates = {
        key: value
        for key, value in features.items()
        if key not in excluded and value > 0
    }
    sorted_items = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
    top = dict(sorted_items[:limit])
    top["sequence_length"] = features.get("sequence_length", 0.0)
    top["approx_molecular_weight"] = features.get("approx_molecular_weight", 0.0)
    return top


def _first_sequence_from_input(sequence_text: str) -> str:
    records = parse_fasta_records(sequence_text)
    if not records:
        return clean_sequence(sequence_text)
    return clean_sequence(records[0].get("sequence_text", ""))


def predict_sequence(sequence_text: str) -> Dict:
    sequence = _first_sequence_from_input(sequence_text)
    if not is_valid_sequence(sequence):
        return {
            "ok": False,
            "error": "Invalid protein sequence. Use at least 30 residues containing only A C D E F G H I K L M N P Q R S T V W Y.",
        }

    artifact = load_model()
    model = artifact["model"] if isinstance(artifact, dict) and "model" in artifact else artifact
    feature_columns = artifact.get("feature_columns", []) if isinstance(artifact, dict) else []
    X = extract_feature_dataframe([sequence])
    if feature_columns:
        X = X.reindex(columns=feature_columns, fill_value=0.0)

    prediction = str(model.predict(X)[0])
    probabilities = _probabilities(model, X)
    if not probabilities:
        probabilities = {
            "cancerous": 1.0 if prediction == "cancerous" else 0.0,
            "non_cancerous": 1.0 if prediction == "non_cancerous" else 0.0,
        }
    probabilities = {
        "cancerous": float(probabilities.get("cancerous", 0.0)),
        "non_cancerous": float(probabilities.get("non_cancerous", 0.0)),
    }
    confidence = max(probabilities.values())
    features = extract_features(sequence)
    return {
        "ok": True,
        "prediction": prediction,
        "display_label": "Cancerous" if prediction == "cancerous" else "Non-Cancerous",
        "confidence": confidence,
        "probabilities": probabilities,
        "sequence": sequence,
        "sequence_length": len(sequence),
        "amino_acid_composition": amino_acid_composition(sequence),
        "frequent_kmers": {
            "k3": top_kmers(sequence, k=3, limit=8),
            "k5": top_kmers(sequence, k=5, limit=8),
        },
        "top_features": _top_features(features),
        "features": features,
        "medical_warning": "Research/demo prediction only. This is not a medical diagnostic result.",
    }


def get_model_status() -> Dict:
    metrics = load_metrics()
    exists = BEST_MODEL_PATH.exists()
    return {
        "model_exists": exists,
        "model_path": str(BEST_MODEL_PATH),
        "last_trained_timestamp": metrics.get("last_trained_timestamp"),
        "best_model_name": metrics.get("best_model_name"),
        "trained_from": metrics.get("trained_from"),
        "metrics": metrics,
    }
