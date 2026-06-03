import math
from collections import Counter
from itertools import product
from typing import Dict, Iterable, List, Set

import pandas as pd

from agent.config import STANDARD_AMINO_ACIDS


AMINO_ACIDS = sorted(STANDARD_AMINO_ACIDS)
DIPEPTIDES = ["".join(pair) for pair in product(AMINO_ACIDS, repeat=2)]
KMER_PATTERN_SIZES = (3, 4, 5)
HASHED_KMER_SPECS = ((3, 256), (4, 256))

MOLECULAR_WEIGHTS = {
    "A": 89.09,
    "C": 121.16,
    "D": 133.10,
    "E": 147.13,
    "F": 165.19,
    "G": 75.07,
    "H": 155.16,
    "I": 131.17,
    "K": 146.19,
    "L": 131.17,
    "M": 149.21,
    "N": 132.12,
    "P": 115.13,
    "Q": 146.15,
    "R": 174.20,
    "S": 105.09,
    "T": 119.12,
    "V": 117.15,
    "W": 204.23,
    "Y": 181.19,
}

HYDROPHOBIC = set("AVILMFWYP")
AROMATIC = set("FWY")
CHARGED = set("DEKRH")
POLAR = set("STNQCY")
BASIC = set("KRH")
ACIDIC = set("DE")


def amino_acid_composition(sequence: str) -> Dict[str, float]:
    length = max(len(sequence), 1)
    return {f"aa_{aa}": sequence.count(aa) / length for aa in AMINO_ACIDS}


def dipeptide_composition(sequence: str) -> Dict[str, float]:
    total = max(len(sequence) - 1, 1)
    counts = {f"dp_{dipeptide}": 0.0 for dipeptide in DIPEPTIDES}
    for index in range(len(sequence) - 1):
        dipeptide = sequence[index : index + 2]
        key = f"dp_{dipeptide}"
        if key in counts:
            counts[key] += 1.0
    return {key: value / total for key, value in counts.items()}


def _stable_hash(value: str) -> int:
    hash_value = 0
    for char in value:
        hash_value = (hash_value * 131 + ord(char)) % 1_000_000_007
    return hash_value


def hashed_kmer_composition(sequence: str, k: int, bins: int) -> Dict[str, float]:
    total = max(len(sequence) - k + 1, 1)
    counts = {f"hk{k}_{index:03d}": 0.0 for index in range(bins)}
    if len(sequence) < k:
        return counts
    for index in range(len(sequence) - k + 1):
        kmer = sequence[index : index + k]
        bucket = _stable_hash(kmer) % bins
        counts[f"hk{k}_{bucket:03d}"] += 1.0
    return {key: value / total for key, value in counts.items()}


def kmer_counts(sequence: str, k: int) -> Counter:
    if k <= 0 or len(sequence) < k:
        return Counter()
    return Counter(sequence[index : index + k] for index in range(len(sequence) - k + 1))


def top_kmers(sequence: str, k: int = 3, limit: int = 10) -> Dict[str, int]:
    return dict(kmer_counts(sequence, k).most_common(limit))


def _normalized_entropy(counts: Counter, total: int) -> float:
    if total <= 1 or not counts:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    max_entropy = math.log2(min(total, max(len(counts), 1)))
    if max_entropy == 0:
        return 0.0
    return entropy / max_entropy


def kmer_pattern_features(sequence: str) -> Dict[str, float]:
    features = {}
    for k in KMER_PATTERN_SIZES:
        counts = kmer_counts(sequence, k)
        total = max(len(sequence) - k + 1, 1)
        repeated_total = sum(count for count in counts.values() if count > 1)
        max_count = max(counts.values(), default=0)
        features.update(
            {
                f"kmer{k}_unique_ratio": len(counts) / total,
                f"kmer{k}_max_frequency_ratio": max_count / total,
                f"kmer{k}_repeated_ratio": repeated_total / total,
                f"kmer{k}_entropy": _normalized_entropy(counts, total),
            }
        )
    return features


def _ratio(sequence: str, residues: Set[str]) -> float:
    length = max(len(sequence), 1)
    return sum(1 for residue in sequence if residue in residues) / length


def sequence_physicochemical_features(sequence: str) -> Dict[str, float]:
    length = len(sequence)
    approximate_molecular_weight = sum(MOLECULAR_WEIGHTS.get(residue, 0.0) for residue in sequence)
    return {
        "sequence_length": float(length),
        "approx_molecular_weight": approximate_molecular_weight,
        "hydrophobic_ratio": _ratio(sequence, HYDROPHOBIC),
        "aromatic_ratio": _ratio(sequence, AROMATIC),
        "charged_ratio": _ratio(sequence, CHARGED),
        "polar_ratio": _ratio(sequence, POLAR),
        "basic_ratio": _ratio(sequence, BASIC),
        "acidic_ratio": _ratio(sequence, ACIDIC),
    }


def extract_features(sequence: str) -> Dict[str, float]:
    features = {}
    features.update(amino_acid_composition(sequence))
    features.update(dipeptide_composition(sequence))
    for k, bins in HASHED_KMER_SPECS:
        features.update(hashed_kmer_composition(sequence, k, bins))
    features.update(kmer_pattern_features(sequence))
    features.update(sequence_physicochemical_features(sequence))
    return features


def extract_feature_dataframe(sequences: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = [extract_features(sequence) for sequence in sequences]
    columns = [f"aa_{aa}" for aa in AMINO_ACIDS]
    columns.extend(f"dp_{dipeptide}" for dipeptide in DIPEPTIDES)
    for k, bins in HASHED_KMER_SPECS:
        columns.extend(f"hk{k}_{index:03d}" for index in range(bins))
    for k in KMER_PATTERN_SIZES:
        columns.extend(
            [
                f"kmer{k}_unique_ratio",
                f"kmer{k}_max_frequency_ratio",
                f"kmer{k}_repeated_ratio",
                f"kmer{k}_entropy",
            ]
        )
    columns.extend(
        [
            "sequence_length",
            "approx_molecular_weight",
            "hydrophobic_ratio",
            "aromatic_ratio",
            "charged_ratio",
            "polar_ratio",
            "basic_ratio",
            "acidic_ratio",
        ]
    )
    return pd.DataFrame(rows, columns=columns).fillna(0.0)
