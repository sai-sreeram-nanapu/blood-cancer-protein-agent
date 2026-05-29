import logging
import warnings
from typing import Dict, List
from urllib.parse import quote_plus

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")

import requests
from Bio import Entrez

from agent.config import ENTREZ_EMAIL, KAGGLE_KEY, KAGGLE_USERNAME


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


def search_ncbi(query: str, max_results: int = 20) -> List[Dict]:
    if not ENTREZ_EMAIL:
        logger.warning("Skipping NCBI search because ENTREZ_EMAIL is not configured.")
        return []

    try:
        Entrez.email = ENTREZ_EMAIL
        with Entrez.esearch(db="protein", term=query, retmax=max_results) as handle:
            search_record = Entrez.read(handle)
        ids = search_record.get("IdList", [])
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
                    notes="NCBI Protein search result",
                )
            )
        return results
    except Exception as exc:  # noqa: BLE001
        logger.warning("NCBI search failed for query '%s': %s", query, exc)
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


def search_uniprot(query: str, max_results: int = 50) -> List[Dict]:
    try:
        params = {
            "query": query,
            "format": "json",
            "size": max_results,
            "fields": "accession,protein_name,organism_name",
        }
        response = requests.get(
            "https://rest.uniprot.org/uniprotkb/search",
            params=params,
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        results = []
        for item in payload.get("results", []):
            accession = item.get("primaryAccession")
            if not accession:
                continue
            organism = item.get("organism", {}).get("scientificName", "")
            title = _uniprot_title(item)
            notes = f"Organism: {organism}" if organism else "UniProtKB search result"
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
        return results
    except Exception as exc:  # noqa: BLE001
        logger.warning("UniProt search failed for query '%s': %s", query, exc)
        return []


def search_zenodo(query: str, max_results: int = 20) -> List[Dict]:
    try:
        response = requests.get(
            "https://zenodo.org/api/records",
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
        return results
    except Exception as exc:  # noqa: BLE001
        logger.warning("Zenodo search failed for query '%s': %s", query, exc)
        return []


def search_kaggle(query: str, max_results: int = 10) -> List[Dict]:
    if not (KAGGLE_USERNAME and KAGGLE_KEY):
        logger.warning("Skipping Kaggle search because Kaggle credentials are not configured.")
        return []

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
        return results
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kaggle search failed for query '%s': %s", query, exc)
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
    """Search compact, label-directed public protein sources for training data."""
    all_results: List[Dict] = []
    seen = set()

    for spec in TARGETED_PUBLIC_QUERIES:
        source = spec["source"]
        query = spec["query"]
        max_results = int(spec["max_results"])
        label_hint = spec["label_hint"]

        if source == "UniProt":
            query_results = search_uniprot(query, max_results=max_results)
        elif source == "NCBI Protein":
            query_results = search_ncbi(query, max_results=max_results)
        else:
            query_results = []

        for result in query_results:
            key = (result.get("source"), result.get("source_id"), result.get("url"))
            if key in seen:
                continue
            seen.add(key)
            result["label_hint"] = label_hint
            result["notes"] = (
                f"{result.get('notes', '')} Label assigned from targeted public-data query: {label_hint}. "
                "Sequence is public-source authentic; label is metadata/query-derived and not clinically validated."
            ).strip()
            all_results.append(result)

    logger.info("Targeted public sequence search completed with %s unique results.", len(all_results))
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
