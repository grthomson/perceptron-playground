# scripts/train_linkage_perceptron.py
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from perceptron.training_utils import build_training_Xy_pairs, train_perceptron_xy

DATA = Path(__file__).resolve().parents[1] / "data"
MODELS = DATA / "models"


def main():
    p = argparse.ArgumentParser(description="Train perceptron for entity resolution")
    p.add_argument("--left", default=str(DATA / "toy_people_clean.csv"))
    p.add_argument("--right", default=str(DATA / "toy_people_noisy.csv"))
    p.add_argument("--labels", default=str(DATA / "toy_labels.csv"))
    p.add_argument("--id-left", default="id")
    p.add_argument("--id-right", default="id_right")
    p.add_argument("--cols", default="forename,surname,address,city,postcode")
    p.add_argument(
        "--neg-ratio", type=float, default=1.0, help="non-matches per actual match"
    )
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--n-iter", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    # outputs
    p.add_argument("--model-out", default=str(MODELS / "linkage_perceptron.pkl"))
    p.add_argument("--pairs-out", default=str(DATA / "train_pairs.csv"))
    p.add_argument("--design-out", default=str(DATA / "train_design.csv"))
    p.add_argument("--config-out", default=str(DATA / "train_config.json"))
    args = p.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)

    left = pd.read_csv(args.left)
    right = pd.read_csv(args.right)
    labels = pd.read_csv(args.labels)
    cols = [c.strip() for c in args.cols.split(",") if c.strip()]

    # Build X, y, and the *exact* pairs used (matches + sampled non-matches)
    X, y, pairs = build_training_Xy_pairs(
        left,
        right,
        labels,
        id_left=args.id_left,
        id_right=args.id_right,
        cols=cols,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
    )

    # Train perceptron
    clf = train_perceptron_xy(X, y, eta=args.eta, n_iter=args.n_iter, seed=args.seed)

    # Save model (include errors_ for learning-curve plot)
    with open(args.model_out, "wb") as f:
        pickle.dump(
            {
                "w_": np.asarray(clf.w_, float),
                "b_": float(clf.b_),
                "errors_": list(clf.errors_),
                "cols": cols,
                "id_left": args.id_left,
                "id_right": args.id_right,
                "eta": args.eta,
                "n_iter": args.n_iter,
                "seed": args.seed,
            },
            f,
        )

    # Save the exact training pairs + labels (for audit/reuse)
    pd.DataFrame(
        {
            "left_id": [p[0] for p in pairs],
            "right_id": [p[1] for p in pairs],
            "label": y,
        }
    ).to_csv(args.pairs_out, index=False)

    # Save the design matrix used for training (features + label)
    feat_cols = [f"{c}_sim" for c in cols]
    pd.DataFrame(X, columns=feat_cols).assign(label=y).to_csv(
        args.design_out, index=False
    )

    # Save a tiny config
    with open(args.config_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cols": cols,
                "id_left": args.id_left,
                "id_right": args.id_right,
                "neg_ratio": args.neg_ratio,
                "seed": args.seed,
            },
            f,
            indent=2,
        )

    print(f"Model:   {args.model_out}")
    print(f"Pairs:   {args.pairs_out}")
    print(f"Design:  {args.design_out}")
    print(f"Config:  {args.config_out}")
    print("Weights:", clf.w_)
    print("Bias:   ", clf.b_)
    print("Updates per epoch:", clf.errors_)


if __name__ == "__main__":
    main()
