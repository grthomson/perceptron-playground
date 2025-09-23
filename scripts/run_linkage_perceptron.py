import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from data_prep.create_linkage_features import build_levenshtein_features

DATA = Path(__file__).resolve().parents[1] / "data"
MODELS = DATA / "models"


def main():
    p = argparse.ArgumentParser(
        description="Run (score) linkage perceptron on candidate pairs"
    )
    p.add_argument("--left", default=str(DATA / "toy_people_clean.csv"))
    p.add_argument("--right", default=str(DATA / "toy_people_noisy.csv"))
    p.add_argument("--model", default=str(MODELS / "linkage_perceptron.pkl"))
    p.add_argument(
        "--candidates", default="", help="CSV with left_id,right_id (optional)"
    )
    p.add_argument("--out", default=str(DATA / "toy_scored_pairs.csv"))
    p.add_argument("--threshold", type=float, default=0.0, help="decision boundary")
    args = p.parse_args()

    left = pd.read_csv(args.left)
    right = pd.read_csv(args.right)

    with open(args.model, "rb") as f:
        m = pickle.load(f)

    cols = m["cols"]
    id_left = m["id_left"]
    id_right = m["id_right"]
    w = np.asarray(m["w_"], dtype=float)
    b = float(m["b_"])

    if args.candidates:
        cand = pd.read_csv(args.candidates)
        pairs = list(zip(cand["left_id"], cand["right_id"], strict=False))
    else:
        # score all pairs (cartesian) — OK for small toy data
        pairs = [
            (l_id, r_id)
            for l_id in left[id_left].tolist()
            for r_id in right[id_right].tolist()
        ]

    X = build_levenshtein_features(left, right, pairs, id_left, id_right, cols)
    scores = X @ w + b
    pred = (scores >= args.threshold).astype(int)

    out = pd.DataFrame(
        {
            "left_id": [p[0] for p in pairs],
            "right_id": [p[1] for p in pairs],
            "score": scores,
            "pred": pred,
        }
    )
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  (rows={len(out)})")


if __name__ == "__main__":
    main()
