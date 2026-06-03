import csv
import json
import logging
import re
import time
import warnings
from typing import Dict, List, Optional, Set
from urllib.parse import quote_plus

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")

import requests
from Bio import Entrez

from agent.api_audit import append_api_audit
from agent.config import (
    ENTREZ_EMAIL,
    FETCH_PAGE_CYCLE_LIMIT,
    FETCH_PAGE_WINDOW,
    FETCH_STATE_PATH,
    KAGGLE_KEY,
    KAGGLE_USERNAME,
    TRAINING_DATA_PATH,
)


logger = logging.getLogger(__name__)

SEARCH_TERMS = [
    "blood cancer protein sequence",
    "leukemia protein sequence dataset",
    "lymphoma protein sequence dataset",
    "myeloma protein sequence dataset",
    "cancer protein FASTA",
    "tumor protein sequence",
    "oncogene protein sequence",
    "cancer associated protein sequence",
    "non cancer protein sequence dataset",
    "healthy protein sequence FASTA",
    "normal human protein sequence",
    "control protein sequence",
]

TARGETED_PUBLIC_QUERIES = [
    {
        "source": "UniProt",
        "query": "leukemia AND organism_id:9606 AND reviewed:true",
        "label_hint": "cancerous",
        "max_results": 40,
    },
    {
        "source": "UniProt",
        "query": "lymphoma AND organism_id:9606 AND reviewed:true",
        "label_hint": "cancerous",
        "max_results": 40,
    },
    {
        "source": "UniProt",
        "query": "myeloma AND organism_id:9606 AND reviewed:true",
        "label_hint": "cancerous",
        "max_results": 30,
    },
    {
        "source": "UniProt",
        "query": "(oncogene OR tumor OR cancer) AND organism_id:9606 AND reviewed:true",
        "label_hint": "cancerous",
        "max_results": 50,
    },
    {
        "source": "UniProt",
        "query": "organism_id:9606 AND reviewed:true AND proteome:up000005640 NOT cancer NOT tumor NOT leukemia NOT lymphoma NOT myeloma NOT oncogene",
        "label_hint": "non_cancerous",
        "max_results": 80,
    },
    {
        "source": "UniProt",
        "query": "organism_id:9606 AND reviewed:true AND keyword:Reference proteome NOT cancer NOT tumor NOT leukemia NOT lymphoma NOT myeloma NOT oncogene",
        "label_hint": "non_cancerous",
        "max_results": 80,
    },
    {
        "source": "NCBI Protein",
        "query": "Homo sapiens[Organism] AND leukemia[All Fields] AND protein[All Fields]",
        "label_hint": "cancerous",
        "max_results": 12,
    },
    {
        "source": "NCBI Protein",
        "query": "Homo sapiens[Organism] AND lymphoma[All Fields] AND protein[All Fields]",
        "label_hint": "cancerous",
        "max_results": 12,
    },
]

CANCER_HINTS = (
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
NON_CANCER_HINTS = (
    "healthy",
    "normal",
    "control",
    "non-cancer",
    "non cancer",
    "reference",
    "normal human",
)


def _source_key(source: str, source_id: str) -> str:
    return f"{source or ''}::{source_id or ''}".lower()


def _known_source_keys() -> Set[str]:
    if not TRAINING_DATA_PATH.exists():
        return set()

    try:
        with TRAINING_DATA_PATH.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return {
                _source_key(row.get("source", ""), row.get("source_id", ""))
                for row in reader
                if row.get("source") and row.get("source_id")
            }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read known training source IDs: %s", exc)
        return set()


def _load_fetch_state() -> Dict[str, int]:
    if not FETCH_STATE_PATH.exists():
        return {}
    try:
        payload = json.loads(FETCH_STATE_PATH.read_text(encoding="utf-8"))
        return {str(key): int(value) for key, value in payload.items()}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read fetch state; starting from first public result pages: %s", exc)
        return {}


def _save_fetch_state(state: Dict[str, int]) -> None:
    try:
        FETCH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FETCH_STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save fetch state: %s", exc)


def _state_key(source: str, query: str) -> str:
    compact_query = re.sub(r"\s+", " ", query.strip())
    return f"{source}:{compact_query}"


def _next_link(link_header: str) -> Optional[str]:
    for part in (link_header or "").split(","):
        section = part.strip()
        if 'rel="next"' not in section:
            continue
        match = re.search(r"<([^>]+)>", section)
        if match:
            return match.group(1)
    return None


def _label_hint(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in NON_CANCER_HINTS):
        return "non_cancerous"
    if any(term in lowered for term in CANCER_HINTS):
        return "cancerous"
    return "unknown"


def _metadata(
    source: str,
    title: str,
    url: str,
    query: str,
    source_id: str,
    notes: str = "",
    **extra,
) -> Dict:
    combined = " ".join([title or "", query or "", notes or ""])
    row = {
        "source": source,
        "title": title or source_id,
        "url": url,
        "query": query,
        "source_id": source_id,
        "label_hint": _label_hint(combined),
        "notes": notes,
    }
    row.update(extra)
    return row


def search_ncbi(query: str, max_results: int = 20, retstart: int = 0) -> List[Dict]:
    if not ENTREZ_EMAIL:
        logger.warning("Skipping NCBI search because ENTREZ_EMAIL is not configured.")
        append_api_audit(
            {
                "stage": "search",
                "source": "NCBI Protein",
                "method": "Entrez.esearch",
                "endpoint": "protein",
                "query": query,
                "status": "skipped",
                "message": "ENTREZ_EMAIL is not configured.",
            }
        )
        return []

    start = time.perf_counter()
    try:
        Entrez.email = ENTREZ_EMAIL
        with Entrez.esearch(db="protein", term=query, retmax=max_results, retstart=retstart) as handle:
            search_record = Entrez.read(handle)
        ids = search_record.get("IdList", [])
        append_api_audit(
            {
                "stage": "search",
                "source": "NCBI Protein",
                "method": "Entrez.esearch",
                "endpoint": "protein",
                "query": query,
                "status": "success",
                "result_count": len(ids),
                "duration_ms": round((time.perf_counter() - start) * 1000),
                "message": f"retstart={retstart}; retmax={max_results}",
            }
        )
        if not ids:
            return []

        titles = {}
        try:
            with Entrez.esummary(db="protein", id=",".join(ids), retmode="xml") as handle:
                summaries = Entrez.read(handle)
            for summary in summaries:
                sid = str(summary.get("Id", ""))
                titles[sid] = summary.get("Title") or summary.get("Caption") or sid
        except Exception as exc:  # noqa: BLE001
            logger.warning("NCBI summary lookup failed for query '%s': %s", query, exc)

        results = []
        for source_id in ids:
            source_id = str(source_id)
            title = titles.get(source_id, f"NCBI protein record {source_id}")
            results.append(
                _metadata(
                    source="NCBI Protein",
                    title=title,
                    url=f"https://www.ncbi.nlm.nih.gov/protein/{source_id}",
                    query=query,
                    source_id=source_id,
                    notes=f"NCBI Protein search result. Result offset: {retstart}.",
                )
            )
        return results
    except Exception as exc:  # noqa: BLE001
        logger.warning("NCBI search failed for query '%s': %s", query, exc)
        append_api_audit(
            {
                "stage": "search",
                "source": "NCBI Protein",
                "method": "Entrez.esearch",
                "endpoint": "protein",
                "query": query,
                "status": "failed",
                "duration_ms": round((time.perf_counter() - start) * 1000),
                "message": str(exc),
            }
        )
        return []


def _uniprot_title(item: Dict) -> str:
    protein = item.get("proteinDescription", {})
    recommended = protein.get("recommendedName", {})
    full_name = recommended.get("fullName", {})
    value = full_name.get("value")
    if value:
        return value
    submitted = protein.get("submissionNames", [])
    if submitted and submitted[0].get("fullName", {}).get("value"):
        return submitted[0]["fullName"]["value"]
    return item.get("primaryAccession", "UniProt protein record")


def search_uniprot(query: str, max_results: int = 50, page_index: int = 0) -> List[Dict]:
    start = time.perf_counter()
    endpoint = "https://rest.uniprot.org/uniprotkb/search"
    try:
        page_size = max(1, min(int(max_results), 500))
        params = {
            "query": query,
            "format": "json",
            "size": page_size,
            "fields": "accession,protein_name,organism_name",
        }
        url = endpoint
        response = None
        for current_page in range(max(0, page_index) + 1):
            response = requests.get(
                url,
                params=params if current_page == 0 else None,
                timeout=20,
            )
            response.raise_for_status()
            if current_page == page_index:
                break
            next_url = _next_link(response.headers.get("Link", ""))
            if not next_url:
                append_api_audit(
                    {
                        "stage": "search",
                        "source": "UniProt",
                        "method": "GET",
                        "endpoint": endpoint,
                        "query": query,
                        "status": "success",
                        "status_code": response.status_code,
                        "result_count": 0,
                        "bytes_received": len(response.content),
                        "duration_ms": round((time.perf_counter() - start) * 1000),
                        "message": f"No next page before requested page_index={page_index}.",
                    }
                )
                return []
            url = next_url
            params = {}

        if response is None:
            return []

        response.raise_for_status()
        payload = response.json()
        results = []
        for item in payload.get("results", []):
            accession = item.get("primaryAccession")
            if not accession:
                continue
            organism = item.get("organism", {}).get("scientificName", "")
            title = _uniprot_title(item)
            organism_note = f"Organism: {organism}" if organism else "UniProtKB search result"
            notes = f"{organism_note}. Result page: {page_index}."
            results.append(
                _metadata(
                    source="UniProt",
                    title=title,
                    url=f"https://www.uniprot.org/uniprotkb/{accession}/entry",
                    query=query,
                    source_id=accession,
                    notes=notes,
                )
            )
        append_api_audit(
            {
                "stage": "search",
                "source": "UniProt",
                "method": "GET",
                "endpoint": endpoint,
                "query": query,
                "status": "success",
                "status_code": response.status_code,
                "result_count": len(results),
                "bytes_received": len(response.content),
                "duration_ms": round((time.perf_counter() - start) * 1000),
                "message": f"page_index={page_index}; page_size={page_size}",
            }
        )
        return results
    except Exception as exc:  # noqa: BLE001
        logger.warning("UniProt search failed for query '%s': %s", query, exc)
        append_api_audit(
            {
                "stage": "search",
                "source": "UniProt",
                "method": "GET",
                "endpoint": endpoint,
                "query": query,
                "status": "failed",
                "duration_ms": round((time.perf_counter() - start) * 1000),
                "message": str(exc),
            }
        )
        return []


def search_zenodo(query: str, max_results: int = 20) -> List[Dict]:
    start = time.perf_counter()
    endpoint = "https://zenodo.org/api/records"
    try:
        response = requests.get(
            endpoint,
            params={"q": query, "size": max_results},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        results = []
        for item in payload.get("hits", {}).get("hits", []):
            record_id = str(item.get("id", ""))
            metadata = item.get("metadata", {})
            title = metadata.get("title", f"Zenodo record {record_id}")
            files = []
            for file_info in item.get("files", []):
                links = file_info.get("links", {})
                files.append(
                    {
                        "key": file_info.get("key", ""),
                        "size": file_info.get("size", 0),
                        "download_url": links.get("self") or links.get("download", ""),
                    }
                )
            results.append(
                _metadata(
                    source="Zenodo",
                    title=title,
                    url=f"https://zenodo.org/records/{record_id}",
                    query=query,
                    source_id=record_id,
                    notes="Zenodo record; downloader will use suitable FASTA, TXT, or CSV files only.",
                    files=files,
                )
            )
        append_api_audit(
            {
                "stage": "search",
                "source": "Zenodo",
                "method": "GET",
                "endpoint": endpoint,
                "query": query,
                "status": "success",
                "status_code": response.status_code,
                "result_count": len(results),
                "bytes_received": len(response.content),
                "duration_ms": round((time.perf_counter() - start) * 1000),
                "message": f"size={max_results}",
            }
        )
        return results
    except Exception as exc:  # noqa: BLE001
        logger.warning("Zenodo search failed for query '%s': %s", query, exc)
        append_api_audit(
            {
                "stage": "search",
                "source": "Zenodo",
                "method": "GET",
                "endpoint": endpoint,
                "query": query,
                "status": "failed",
                "duration_ms": round((time.perf_counter() - start) * 1000),
                "message": str(exc),
            }
        )
        return []


def search_kaggle(query: str, max_results: int = 10) -> List[Dict]:
    if not (KAGGLE_USERNAME and KAGGLE_KEY):
        logger.warning("Skipping Kaggle search because Kaggle credentials are not configured.")
        append_api_audit(
            {
                "stage": "search",
                "source": "Kaggle",
                "method": "KaggleApi.dataset_list",
                "endpoint": "kaggle datasets",
                "query": query,
                "status": "skipped",
                "message": "Kaggle credentials are not configured.",
            }
        )
        return []

    start = time.perf_counter()
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        datasets = api.dataset_list(search=query, sort_by="hottest")
        results = []
        for dataset in datasets[:max_results]:
            ref = getattr(dataset, "ref", "")
            title = getattr(dataset, "title", ref)
            subtitle = getattr(dataset, "subtitle", "")
            results.append(
                _metadata(
                    source="Kaggle",
                    title=title,
                    url=f"https://www.kaggle.com/datasets/{ref}",
                    query=query,
                    source_id=ref,
                    notes=subtitle or "Kaggle dataset search result",
                )
            )
        append_api_audit(
            {
                "stage": "search",
                "source": "Kaggle",
                "method": "KaggleApi.dataset_list",
                "endpoint": "kaggle datasets",
                "query": query,
                "status": "success",
                "result_count": len(results),
                "duration_ms": round((time.perf_counter() - start) * 1000),
                "message": f"max_results={max_results}",
            }
        )
        return results
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kaggle search failed for query '%s': %s", query, exc)
        append_api_audit(
            {
                "stage": "search",
                "source": "Kaggle",
                "method": "KaggleApi.dataset_list",
                "endpoint": "kaggle datasets",
                "query": query,
                "status": "failed",
                "duration_ms": round((time.perf_counter() - start) * 1000),
                "message": str(exc),
            }
        )
        return []


def run_dataset_search() -> List[Dict]:
    all_results: List[Dict] = []
    seen = set()
    for query in SEARCH_TERMS:
        query_results = []
        query_results.extend(search_ncbi(query))
        query_results.extend(search_uniprot(query))
        query_results.extend(search_zenodo(query))
        query_results.extend(search_kaggle(query))
        for result in query_results:
            key = (result.get("source"), result.get("source_id"), result.get("url"))
            if key in seen:
                continue
            seen.add(key)
            all_results.append(result)
    logger.info("Dataset search completed with %s unique results.", len(all_results))
    return all_results


def run_targeted_public_sequence_search() -> List[Dict]:
    """Search label-directed public protein sources while avoiding known records."""
    all_results: List[Dict] = []
    seen = set()
    known_keys = _known_source_keys()
    fetch_state = _load_fetch_state()

    for spec in TARGETED_PUBLIC_QUERIES:
        source = spec["source"]
        query = spec["query"]
        max_results = int(spec["max_results"])
        label_hint = spec["label_hint"]
        state_key = _state_key(source, query)
        page_cycle_limit = max(1, int(FETCH_PAGE_CYCLE_LIMIT))
        start_page = max(0, int(fetch_state.get(state_key, 0))) % page_cycle_limit
        pages_checked = 0
        query_new_results = 0

        for page_offset in range(FETCH_PAGE_WINDOW):
            page_index = start_page + page_offset
            pages_checked += 1

            if source == "UniProt":
                query_results = search_uniprot(query, max_results=max_results, page_index=page_index)
            elif source == "NCBI Protein":
                query_results = search_ncbi(
                    query,
                    max_results=max_results,
                    retstart=page_index * max_results,
                )
            else:
                query_results = []

            if not query_results:
                break

            for result in query_results:
                source_record_key = _source_key(result.get("source", ""), result.get("source_id", ""))
                key = (result.get("source"), result.get("source_id"), result.get("url"))
                if source_record_key in known_keys or key in seen:
                    continue
                seen.add(key)
                result["label_hint"] = label_hint
                result["notes"] = (
                    f"{result.get('notes', '')} Label assigned from targeted public-data query: {label_hint}. "
                    "Sequence is public-source authentic; label is metadata/query-derived and not clinically validated. "
                    "Already-seen public source IDs are skipped before download."
                ).strip()
                all_results.append(result)
                query_new_results += 1
                if query_new_results >= max_results:
                    break

            if query_new_results >= max_results:
                break
            if len(query_results) < max_results:
                break

        if pages_checked:
            fetch_state[state_key] = (start_page + pages_checked) % page_cycle_limit

    _save_fetch_state(fetch_state)
    logger.info(
        "Targeted public sequence search completed with %s unique new results; skipped %s known source IDs.",
        len(all_results),
        len(known_keys),
    )
    return all_results


def source_search_url(source: str, query: str) -> str:
    encoded = quote_plus(query)
    if source == "NCBI Protein":
        return f"https://www.ncbi.nlm.nih.gov/protein/?term={encoded}"
    if source == "UniProt":
        return f"https://www.uniprot.org/uniprotkb?query={encoded}"
    if source == "Zenodo":
        return f"https://zenodo.org/search?q={encoded}"
    if source == "Kaggle":
        return f"https://www.kaggle.com/datasets?search={encoded}"
    return ""
