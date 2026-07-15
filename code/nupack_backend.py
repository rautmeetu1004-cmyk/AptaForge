#!/usr/bin/env python
"""Thin DNA folding backend used to generate the folding labels shipped in
``data/folding_labels.csv``.

`compute_nupack_features` returns the minimum-free-energy (MFE), ensemble defect,
log partition function and a length-normalised structural entropy for an ssDNA
sequence, computed with NUPACK 4 (``material='dna'``, 37 C by default).

NUPACK is a third-party thermodynamics package that must be installed separately
(https://www.nupack.org). It is *not* redistributed here: only the numeric labels
it produced are shipped, so the rest of the pipeline reproduces without it. Install
NUPACK to regenerate the labels from scratch (``code/models/00_generate_folding_labels.py``).
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

try:
    import nupack  # type: ignore
    from nupack import Model, mfe, defect, pfunc, pairs  # type: ignore
    _HAS_NUPACK = True
except Exception:
    _HAS_NUPACK = False

DNA_BASES = ("A", "T", "G", "C")


def nupack_available() -> bool:
    return _HAS_NUPACK


def _validate_dna(sequence: str) -> str:
    s = sequence.upper()
    if "U" in s:
        raise ValueError("Uracil present: calls here are DNA-only (material='dna').")
    bad = set(s) - set(DNA_BASES)
    if bad:
        raise ValueError(f"Non-DNA characters {sorted(bad)} — reject before folding.")
    return s


def compute_nupack_features(
    sequence: str, temperature: float = 37.0, sodium: float = 1.0, magnesium: float = 0.0
) -> Dict[str, float]:
    """NUPACK 4 DNA path (verified against nupack 4.0.2.1).

    ``sodium``/``magnesium`` (mol/L) drive NUPACK's built-in salt correction; the
    labels shipped here use the package defaults (sodium=1.0, magnesium=0.0).
    """
    if not _HAS_NUPACK:
        raise RuntimeError("NUPACK is not installed; install it to compute folding features.")
    sequence = _validate_dna(sequence)
    model = Model(material="dna", celsius=temperature, sodium=sodium, magnesium=magnesium)
    strand = nupack.Strand(sequence, name="aptamer")

    mfe_result = mfe([strand], model=model)
    mfe_energy = float(mfe_result[0].energy)
    target_structure = mfe_result[0].structure

    ensemble_defect = float(defect(str(target_structure), [sequence], model=model))

    q = pfunc([strand], model=model)[0]
    log_pfunc = float(q.ln()) if hasattr(q, "ln") else math.log(float(q))

    # length-normalised structural entropy from the base-pair probability matrix
    bp = pairs([strand], model=model).to_array()
    flat = bp.flatten()
    flat = flat[flat > 0]
    structural_entropy = float(-np.sum(flat * np.log2(flat + 1e-10)) / max(len(sequence), 1))

    return {
        "mfe_kcal": mfe_energy,
        "ensemble_defect": ensemble_defect,
        "log_pfunc": log_pfunc,
        "structural_entropy": structural_entropy,
        "mfe_structure": str(target_structure),
    }
