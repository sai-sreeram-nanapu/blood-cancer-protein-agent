from itertools import product
from typing import Dict, Iterable, List, Set

import pandas as pd

from agent.config import STANDARD_AMINO_ACIDS


AMINO_ACIDS = sorted(STANDARD_AMINO_ACIDS)
DIPEPTIDES = ["".join(pair) for pair in product(AMINO_ACIDS, repeat=2)]

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
    features.update(sequence_physicochemical_features(sequence))
    return features


def extract_feature_dataframe(sequences: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = [extract_features(sequence) for sequence in sequences]
    columns = [f"aa_{aa}" for aa in AMINO_ACIDS]
    columns.extend(f"dp_{dipeptide}" for dipeptide in DIPEPTIDES)
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
