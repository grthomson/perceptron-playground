# scripts/plot_entity_resolution.py
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from viz.plot import (
    plot_decision_plane_3d,
    plot_decision_regions_2d,
    plot_learning_curve,
)

DATA = Path(__file__).resolve().parents[1] / "data"
MODEL = DATA / "models" / "linkage_perceptron.pkl"
DESIGN = DATA / "train_design.csv"  # produced by train_linkage_perceptron.py


class LoadedPerceptron:
    def __init__(self, w, b, errors):
        self.w_ = np.asarray(w, float)
        self.b_ = float(b)
        self.errors_ = list(errors) if errors is not None else []

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X @ self.w_ + self.b_ >= 0.0).astype(int)


def main() -> None:
    # load trained model (dict with w_, b_, errors_, cols, ...)
    with open(MODEL, "rb") as f:
        m = pickle.load(f)

    # load design matrix used for training (features + label)
    df = pd.read_csv(DESIGN)
    if "label" not in df.columns:
        raise SystemExit(f"'label' column missing from {DESIGN}")

    y = df["label"].to_numpy(dtype=int)
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].to_numpy(dtype=float)

    clf = LoadedPerceptron(m["w_"], m["b_"], m.get("errors_", []))

    # always: learning curve
    if clf.errors_:
        plot_learning_curve(clf.errors_)

    # conditional boundary plots
    n_features = X.shape[1]
    if n_features == 2:
        plot_decision_regions_2d(
            X,
            y,
            classifier=clf,
            feat_idx=(0, 1),
            feature_names=(feature_cols[0], feature_cols[1]),
        )
    elif n_features == 3:
        plot_decision_plane_3d(
            X,
            y,
            classifier=clf,
            feat_idx=(0, 1, 2),
            feature_names=(feature_cols[0], feature_cols[1], feature_cols[2]),
        )
    else:
        print(
            f"Skipping boundary plot: model has {n_features} features (>3). "
            "Consider PCA if you want a projection plot."
        )


if __name__ == "__main__":
    main()
