# src/data_prep/sampling.py
from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import product

import numpy as np
import pandas as pd


def anti_join_nonmatches_pd(
    left: pd.DataFrame,
    right: pd.DataFrame,
    id_left: str,
    id_right: str,
    exclude: Iterable[tuple[object, object]],
    forbid_same_id: bool = False,
) -> pd.DataFrame:
    """
    Pandas anti-join: build the Cartesian product (left × right), then remove
    known matches (exclude). Returns a DataFrame with columns: left_id, right_id.

    NOTE: This materialises N*M rows. Only use for small tables.
    """
    # 1) Cartesian product
    pairs = left[[id_left]].merge(right[[id_right]], how="cross")
    pairs = pairs.rename(columns={id_left: "left_id", id_right: "right_id"})

    # 2) Remove known matches
    excl = pd.DataFrame(exclude, columns=["left_id", "right_id"])
    if not excl.empty:
        pairs = pairs.merge(
            excl, on=["left_id", "right_id"], how="left", indicator=True
        )
        pairs = pairs.loc[pairs["_merge"] == "left_only", ["left_id", "right_id"]]
    # 3) Optionally forbid left_id == right_id
    if forbid_same_id:
        pairs = pairs.loc[pairs["left_id"] != pairs["right_id"]]
    return pairs.reset_index(drop=True)


def all_nonmatches(
    left: pd.DataFrame,
    right: pd.DataFrame,
    id_left: str,
    id_right: str,
    exclude: Iterable[tuple[object, object]],
    forbid_same_id: bool = False,
) -> list[tuple[object, object]]:
    """Python version (no pandas cross) for tiny tables."""
    left_ids = left[id_left].tolist()
    right_ids = right[id_right].tolist()
    exclude_set = set(exclude)

    out: list[tuple[object, object]] = []
    for li, ri in product(left_ids, right_ids):
        if (li, ri) in exclude_set:
            continue
        if forbid_same_id and (li == ri):
            continue
        out.append((li, ri))
    return out


def sample_nonmatches(
    left: pd.DataFrame,
    right: pd.DataFrame,
    id_left: str,
    id_right: str,
    exclude: Sequence[tuple[object, object]],
    k: int,
    seed: int = 1,
    forbid_same_id: bool = False,
    cross_threshold: int = 200_000,
) -> list[tuple[object, object]]:
    """
    Sample k non-matching pairs.

    Strategy:
      - If N*M <= cross_threshold → use pandas anti-join, then sample from it.
      - Otherwise → rejection sampling (no full cross product).
    """
    rng = np.random.default_rng(seed)
    exclude_set = set(exclude)

    n_left = len(left)
    n_right = len(right)
    total = n_left * n_right

    if total <= cross_threshold:
        # Safe to materialise
        candidates = anti_join_nonmatches_pd(
            left, right, id_left, id_right, exclude_set, forbid_same_id=forbid_same_id
        )
        if candidates.empty:
            return []
        if k >= len(candidates):
            # sample all (or return all)
            return list(map(tuple, candidates[["left_id", "right_id"]].to_numpy()))
        idx = rng.choice(len(candidates), size=k, replace=False)
        sub = candidates.iloc[idx][["left_id", "right_id"]].to_numpy()
        return list(map(tuple, sub))

    # Large case: rejection sampling
    left_ids = left[id_left].to_numpy()
    right_ids = right[id_right].to_numpy()
    samples: set[tuple[object, object]] = set()
    attempts = 0
    max_attempts = max(20 * k, 10_000)

    while len(samples) < k and attempts < max_attempts:
        li = rng.choice(left_ids)
        ri = rng.choice(right_ids)
        pair = (li, ri)
        if pair in exclude_set:
            attempts += 1
            continue
        if forbid_same_id and (li == ri):
            attempts += 1
            continue
        samples.add(pair)
        attempts += 1

    # If we struggled to fill k, last resort: take what we have.
    return list(samples)
