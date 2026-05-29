import logging
import re
from typing import Dict, Iterable, List, Optional

import pandas as pd

from agent.config import MIN_SEQUENCE_LENGTH, STANDARD_AMINO_ACIDS, TRAINING_DATA_PATH


logger = logging.getLogger(__name__)

CANCER_LABEL_TERMS = (
    "leukemia",
    "lymphoma",
    "myeloma",
    "blood cancer",
    "cancer",
    "tumor",
    "oncogene",
    "malignancy",
    "cancer-associated",
    "cancer associated",
)

NON_CANCER_LABEL_TERMS = (
    "healthy",
    "normal human",
    "normal",
    "control",
    "non-cancer",
    "non cancer",
    "reference",
)


def clean_sequence(text: str) -> str:
    if not text:
        return ""
    sequence_lines = []
    for line in str(text).splitlines():
        if line.strip().startswith(">"):
            continue
        sequence_lines.append(line)
    cleaned = "".join(sequence_lines).upper()
    cleaned = re.sub(r"[\s\d\-_*.,;:|/\\]+", "", cleaned)
    cleaned = re.sub(r"[^A-Z]", "", cleaned)
    if not cleaned:
        return ""
    if any(residue not in STANDARD_AMINO_ACIDS for residue in cleaned):
        return ""
    return cleaned


def is_valid_sequence(sequence: str) -> bool:
    return (
        bool(sequence)
        and len(sequence) >= MIN_SEQUENCE_LENGTH
        and all(residue in STANDARD_AMINO_ACIDS for residue in sequence)
    )


def parse_fasta_records(text: str) -> List[Dict]:
    text = text or ""
    if ">" not in text:
        return [{"header": "", "sequence_text": text}]

    records = []
    header = ""
    sequence_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            if sequence_lines:
                records.append({"header": header, "sequence_text": "\n".join(sequence_lines)})
            header = stripped[1:].strip()
            sequence_lines = []
        else:
            sequence_lines.append(stripped)
    if sequence_lines:
        records.append({"header": header, "sequence_text": "\n".join(sequence_lines)})
    return records


def infer_label_from_metadata(metadata: Dict) -> Optional[str]:
    label_hint = str(metadata.get("label_hint", "")).lower()
    if label_hint in {"cancerous", "non_cancerous"}:
        return label_hint

    fields = [
        metadata.get("title", ""),
        metadata.get("query", ""),
        metadata.get("notes", ""),
        metadata.get("source", ""),
        metadata.get("header", ""),
    ]
    combined = " ".join(str(field) for field in fields).lower()

    if any(term in combined for term in NON_CANCER_LABEL_TERMS):
        return "non_cancerous"
    if any(term in combined for term in CANCER_LABEL_TERMS):
        return "cancerous"
    return None


def clean_and_label_records(records: Iterable[Dict]) -> pd.DataFrame:
    rows = []
    seen_sequences = set()
    skipped = {"invalid_sequence": 0, "unclear_label": 0, "duplicate": 0}

    for record in records:
        metadata = dict(record.get("metadata", {}))
        for parsed in parse_fasta_records(record.get("text", "")):
            sequence = clean_sequence(parsed.get("sequence_text", ""))
            if not is_valid_sequence(sequence):
                skipped["invalid_sequence"] += 1
                continue
            if sequence in seen_sequences:
                skipped["duplicate"] += 1
                continue
            enriched_metadata = {**metadata, "header": parsed.get("header", "")}
            label = infer_label_from_metadata(enriched_metadata)
            if label is None:
                skipped["unclear_label"] += 1
                continue
            seen_sequences.add(sequence)
            rows.append(
                {
                    "sequence": sequence,
                    "label": label,
                    "source": enriched_metadata.get("source", ""),
                    "title": enriched_metadata.get("title", ""),
                    "url": enriched_metadata.get("url", ""),
                    "query": enriched_metadata.get("query", ""),
                    "source_id": enriched_metadata.get("source_id", ""),
                    "label_hint": enriched_metadata.get("label_hint", ""),
                    "notes": enriched_metadata.get("notes", ""),
                }
            )

    logger.info("Cleaning complete: %s kept, skipped=%s", len(rows), skipped)
    return pd.DataFrame(rows)


def save_training_data(df: pd.DataFrame) -> None:
    TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRAINING_DATA_PATH, index=False)


def merge_with_existing_training_data(new_df: pd.DataFrame) -> Dict:
    TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    previous_df = pd.DataFrame()
    if TRAINING_DATA_PATH.exists():
        try:
            previous_df = pd.read_csv(TRAINING_DATA_PATH)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read existing training data for merge: %s", exc)
            previous_df = pd.DataFrame()

    if new_df is None:
        new_df = pd.DataFrame()

    previous_count = int(len(previous_df))
    new_count = int(len(new_df))
    combined = pd.concat([previous_df, new_df], ignore_index=True, sort=False)

    if combined.empty:
        save_training_data(combined)
        return {
            "training_data": combined,
            "previous_samples": previous_count,
            "new_samples": new_count,
            "merged_samples": 0,
            "duplicates_removed": 0,
            "conflicting_sequences_removed": 0,
        }

    combined = combined.dropna(subset=["sequence", "label"]).copy()
    combined["sequence"] = combined["sequence"].astype(str).str.upper()
    combined["label"] = combined["label"].astype(str)
    combined = combined[combined["sequence"].map(is_valid_sequence)]
    combined = combined[combined["label"].isin(["cancerous", "non_cancerous"])]

    conflict_mask = combined.groupby("sequence")["label"].transform("nunique") > 1
    conflicting_sequences_removed = int(combined.loc[conflict_mask, "sequence"].nunique())
    combined = combined.loc[~conflict_mask].copy()

    before_dedupe = int(len(combined))
    combined = combined.drop_duplicates(subset=["sequence"], keep="first").reset_index(drop=True)
    duplicates_removed = int(before_dedupe - len(combined))

    save_training_data(combined)
    return {
        "training_data": combined,
        "previous_samples": previous_count,
        "new_samples": new_count,
        "merged_samples": int(len(combined)),
        "duplicates_removed": duplicates_removed,
        "conflicting_sequences_removed": conflicting_sequences_removed,
    }
