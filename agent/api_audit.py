import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from agent.config import API_AUDIT_LOG_PATH


FIELDNAMES = [
    "timestamp_utc",
    "stage",
    "source",
    "method",
    "endpoint",
    "query",
    "status",
    "status_code",
    "result_count",
    "bytes_received",
    "duration_ms",
    "message",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_log(path: Path = API_AUDIT_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        reset_api_audit(path)


def reset_api_audit(path: Path = API_AUDIT_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()


def append_api_audit(row: Dict, path: Path = API_AUDIT_LOG_PATH) -> None:
    _ensure_log(path)
    clean_row = {
        "timestamp_utc": row.get("timestamp_utc") or _now(),
        "stage": row.get("stage", ""),
        "source": row.get("source", ""),
        "method": row.get("method", ""),
        "endpoint": row.get("endpoint", ""),
        "query": row.get("query", ""),
        "status": row.get("status", ""),
        "status_code": row.get("status_code", ""),
        "result_count": row.get("result_count", ""),
        "bytes_received": row.get("bytes_received", ""),
        "duration_ms": row.get("duration_ms", ""),
        "message": row.get("message", ""),
    }
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writerow(clean_row)


def load_api_audit(limit: int = 100, path: Path = API_AUDIT_LOG_PATH) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit and limit > 0:
        return rows[-limit:]
    return rows


def summarize_api_audit(rows: List[Dict]) -> Dict:
    total = len(rows)
    successful = sum(1 for row in rows if row.get("status") == "success")
    failed = sum(1 for row in rows if row.get("status") == "failed")
    skipped = sum(1 for row in rows if row.get("status") == "skipped")
    by_source: Dict[str, int] = {}
    for row in rows:
        source = row.get("source") or "unknown"
        by_source[source] = by_source.get(source, 0) + 1
    return {
        "api_calls_logged": total,
        "api_successful_calls": successful,
        "api_failed_calls": failed,
        "api_skipped_calls": skipped,
        "api_calls_by_source": by_source,
    }
