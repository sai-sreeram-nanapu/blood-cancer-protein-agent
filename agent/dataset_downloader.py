import csv
import logging
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")

import requests
from Bio import Entrez

from agent.config import (
    DATA_RAW_DIR,
    DATASET_LOG_PATH,
    ENTREZ_EMAIL,
    KAGGLE_KEY,
    KAGGLE_USERNAME,
    MAX_DOWNLOAD_SIZE_MB,
    ensure_directories,
)


logger = logging.getLogger(__name__)

ALLOWED_DOWNLOAD_EXTENSIONS = {".fasta", ".fa", ".faa", ".txt", ".csv"}


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "record")
    return cleaned[:120].strip("_") or "record"


def _max_bytes() -> int:
    return MAX_DOWNLOAD_SIZE_MB * 1024 * 1024


def _save_raw_file(filename: str, content: str) -> Path:
    ensure_directories()
    path = DATA_RAW_DIR / filename
    path.write_text(content, encoding="utf-8", errors="replace")
    return path


def _record_from_text(text: str, metadata: Dict, file_path: Path) -> Dict:
    return {
        "text": text,
        "metadata": metadata,
        "file_path": str(file_path),
    }


def _log_row(metadata: Dict, status: str, message: str, file_path: str = "", bytes_downloaded: int = 0) -> Dict:
    return {
        "source": metadata.get("source", ""),
        "source_id": metadata.get("source_id", ""),
        "title": metadata.get("title", ""),
        "query": metadata.get("query", ""),
        "url": metadata.get("url", ""),
        "status": status,
        "message": message,
        "file_path": file_path,
        "bytes_downloaded": bytes_downloaded,
    }


def download_ncbi_records(results: Iterable[Dict]) -> Tuple[List[Dict], List[Dict]]:
    records: List[Dict] = []
    log_rows: List[Dict] = []
    ncbi_results = [row for row in results if row.get("source") == "NCBI Protein"]

    if not ENTREZ_EMAIL:
        for metadata in ncbi_results:
            log_rows.append(_log_row(metadata, "skipped", "ENTREZ_EMAIL is not configured."))
        return records, log_rows

    Entrez.email = ENTREZ_EMAIL
    for metadata in ncbi_results:
        source_id = metadata.get("source_id", "")
        try:
            with Entrez.efetch(db="protein", id=source_id, rettype="fasta", retmode="text") as handle:
                text = handle.read()
            size = len(text.encode("utf-8"))
            if size > _max_bytes():
                log_rows.append(_log_row(metadata, "skipped", "NCBI FASTA response exceeded size limit.", bytes_downloaded=size))
                continue
            if not text.strip():
                log_rows.append(_log_row(metadata, "skipped", "NCBI returned an empty FASTA response."))
                continue
            filename = f"ncbi_{_safe_name(source_id)}.fasta"
            path = _save_raw_file(filename, text)
            records.append(_record_from_text(text, metadata, path))
            log_rows.append(_log_row(metadata, "downloaded", "Downloaded NCBI FASTA.", str(path), size))
        except Exception as exc:  # noqa: BLE001
            logger.warning("NCBI download failed for %s: %s", source_id, exc)
            log_rows.append(_log_row(metadata, "failed", str(exc)))
    return records, log_rows


def download_uniprot_records(results: Iterable[Dict]) -> Tuple[List[Dict], List[Dict]]:
    records: List[Dict] = []
    log_rows: List[Dict] = []
    for metadata in [row for row in results if row.get("source") == "UniProt"]:
        accession = metadata.get("source_id", "")
        url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            text = response.text
            size = len(response.content)
            if size > _max_bytes():
                log_rows.append(_log_row(metadata, "skipped", "UniProt FASTA exceeded size limit.", bytes_downloaded=size))
                continue
            if not text.strip():
                log_rows.append(_log_row(metadata, "skipped", "UniProt returned an empty FASTA response."))
                continue
            filename = f"uniprot_{_safe_name(accession)}.fasta"
            path = _save_raw_file(filename, text)
            records.append(_record_from_text(text, metadata, path))
            log_rows.append(_log_row(metadata, "downloaded", "Downloaded UniProt FASTA.", str(path), size))
        except Exception as exc:  # noqa: BLE001
            logger.warning("UniProt download failed for %s: %s", accession, exc)
            log_rows.append(_log_row(metadata, "failed", str(exc)))
    return records, log_rows


def _is_suitable_zenodo_file(file_info: Dict) -> bool:
    key = file_info.get("key", "")
    extension = Path(key).suffix.lower()
    size = int(file_info.get("size") or 0)
    return extension in ALLOWED_DOWNLOAD_EXTENSIONS and 0 < size <= _max_bytes()


def download_zenodo_records(results: Iterable[Dict]) -> Tuple[List[Dict], List[Dict]]:
    records: List[Dict] = []
    log_rows: List[Dict] = []
    for metadata in [row for row in results if row.get("source") == "Zenodo"]:
        files = metadata.get("files", [])
        suitable_files = [file_info for file_info in files if _is_suitable_zenodo_file(file_info)]
        if not suitable_files:
            log_rows.append(_log_row(metadata, "skipped", "No suitable small FASTA, TXT, or CSV files found."))
            continue
        for file_info in suitable_files:
            download_url = file_info.get("download_url", "")
            filename = file_info.get("key", "zenodo_file.txt")
            try:
                response = requests.get(download_url, timeout=30)
                response.raise_for_status()
                size = len(response.content)
                if size > _max_bytes():
                    log_rows.append(_log_row(metadata, "skipped", f"{filename} exceeded size limit.", bytes_downloaded=size))
                    continue
                text = response.text
                path = _save_raw_file(f"zenodo_{_safe_name(metadata.get('source_id', 'record'))}_{_safe_name(filename)}", text)
                records.append(_record_from_text(text, metadata, path))
                log_rows.append(_log_row(metadata, "downloaded", f"Downloaded Zenodo file {filename}.", str(path), size))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Zenodo download failed for %s: %s", filename, exc)
                log_rows.append(_log_row(metadata, "failed", str(exc)))
    return records, log_rows


def download_kaggle_records(results: Iterable[Dict]) -> Tuple[List[Dict], List[Dict]]:
    records: List[Dict] = []
    log_rows: List[Dict] = []
    kaggle_results = [row for row in results if row.get("source") == "Kaggle"]

    if not (KAGGLE_USERNAME and KAGGLE_KEY):
        for metadata in kaggle_results:
            log_rows.append(_log_row(metadata, "skipped", "Kaggle credentials are not configured."))
        return records, log_rows

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:  # noqa: BLE001
        for metadata in kaggle_results:
            log_rows.append(_log_row(metadata, "failed", f"Kaggle API import failed: {exc}"))
        return records, log_rows

    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as exc:  # noqa: BLE001
        for metadata in kaggle_results:
            log_rows.append(_log_row(metadata, "failed", f"Kaggle authentication failed: {exc}"))
        return records, log_rows

    for metadata in kaggle_results:
        ref = metadata.get("source_id", "")
        dataset_dir = DATA_RAW_DIR / f"kaggle_{_safe_name(ref)}"
        try:
            dataset_dir.mkdir(parents=True, exist_ok=True)
            api.dataset_download_files(ref, path=str(dataset_dir), unzip=True, quiet=True)
            found_files = []
            for path in dataset_dir.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in ALLOWED_DOWNLOAD_EXTENSIONS:
                    continue
                size = path.stat().st_size
                if size > _max_bytes():
                    log_rows.append(_log_row(metadata, "skipped", f"{path.name} exceeded size limit.", str(path), size))
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                records.append(_record_from_text(text, metadata, path))
                found_files.append(path.name)
                log_rows.append(_log_row(metadata, "downloaded", f"Read Kaggle file {path.name}.", str(path), size))
            if not found_files:
                log_rows.append(_log_row(metadata, "skipped", "No suitable FASTA, TXT, or CSV files found after Kaggle download."))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Kaggle download failed for %s: %s", ref, exc)
            log_rows.append(_log_row(metadata, "failed", str(exc)))
    return records, log_rows


def download_all(search_results: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    records: List[Dict] = []
    log_rows: List[Dict] = []
    for downloader in (
        download_ncbi_records,
        download_uniprot_records,
        download_zenodo_records,
        download_kaggle_records,
    ):
        downloaded, rows = downloader(search_results)
        records.extend(downloaded)
        log_rows.extend(rows)
    save_dataset_log(log_rows)
    return records, log_rows


def save_dataset_log(log_rows: List[Dict]) -> None:
    DATASET_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "source_id",
        "title",
        "query",
        "url",
        "status",
        "message",
        "file_path",
        "bytes_downloaded",
    ]
    with DATASET_LOG_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in log_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
