from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from data_prep.create_linkage_features import build_levenshtein_features
from data_prep.sampling import sample_nonmatches


def build_training_Xy_pairs(
    left: pd.DataFrame,
    right: pd.DataFrame,
    labels: pd.DataFrame,
    id_left: str,
    id_right: str,
    cols: Sequence[str],
    neg_ratio: float = 1.0,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[tuple[object, object]]]:
    """Return X, y, and the exact (left_id, right_id) pairs used"""
    """(matches + sampled non-matches)."""
    pos_pairs = list(zip(labels["left_id"], labels["right_id"], strict=False))
    n_pos = len(pos_pairs)
    n_neg = int(round(neg_ratio * n_pos))

    neg_pairs = sample_nonmatches(
        left, right, id_left, id_right, exclude=pos_pairs, k=n_neg, seed=seed
    )

    all_pairs = pos_pairs + neg_pairs
    X = build_levenshtein_features(
        left, right, all_pairs, id_left, id_right, list(cols)
    )
    y = np.concatenate(
        [np.ones(len(pos_pairs), dtype=int), np.zeros(len(neg_pairs), dtype=int)]
    )
    return X, y, all_pairs


def train_perceptron_xy(
    X: np.ndarray,
    y: np.ndarray,
    eta: float = 0.1,
    n_iter: int = 10,
    seed: int = 1,
):
    from perceptron.perceptron_class import Perceptron

    clf = Perceptron(eta=eta, n_iter=n_iter, random_state=seed).fit(X, y)
    return clf
