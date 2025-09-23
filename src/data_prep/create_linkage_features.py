from __future__ import annotations

import numpy as np
import pandas as pd

from data_prep.levenshtein import normalized_levenshtein


def build_levenshtein_features(
    left: pd.DataFrame,
    right: pd.DataFrame,
    pairs: list[tuple[object, object]],
    id_left: str,
    id_right: str,
    cols: list[str],
) -> np.ndarray:
    lidx = left.set_index(id_left)
    ridx = right.set_index(id_right)
    rows: list[list[float]] = []
    for lid, rid in pairs:
        lrow, rrow = lidx.loc[lid], ridx.loc[rid]
        rows.append([normalized_levenshtein(lrow[c], rrow[c]) for c in cols])
    return np.asarray(rows, dtype=float)
