import argparse
import pickle
from pathlib import Path

import pandas as pd

from data_prep.create_linkage_features import build_levenshtein_features
from perceptron.perceptron_class import Perceptron

DATA = Path(__file__).resolve().parents[1] / "data"
MODELS = DATA / "models"


def main():
    p = argparse.ArgumentParser(description="Train perceptron for record linkage")
    p.add_argument("--left", default=str(DATA / "toy_people_clean.csv"))
    p.add_argument("--right", default=str(DATA / "toy_people_noisy.csv"))
    p.add_argument("--labels", default=str(DATA / "toy_labels.csv"))
    p.add_argument("--id-left", default="id")
    p.add_argument("--id-right", default="id_right")
    p.add_argument("--cols", default="forename,surname,address,city,postcode")
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--n-iter", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=str(MODELS / "linkage_perceptron.pkl"))
    args = p.parse_args()

    left = pd.read_csv(args.left)
    right = pd.read_csv(args.right)
    labels = pd.read_csv(args.labels)

    cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    pairs = list(zip(labels["left_id"], labels["right_id"], strict=False))
    y = labels["label"].astype(int).to_numpy()

    X = build_levenshtein_features(
        left, right, pairs, args.id_left, args.id_right, cols
    )

    clf = Perceptron(eta=args.eta, n_iter=args.n_iter, random_state=args.seed).fit(X, y)

    MODELS.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(
            {
                "w_": clf.w_,
                "b_": clf.b_,
                "cols": cols,
                "id_left": args.id_left,
                "id_right": args.id_right,
                "eta": args.eta,
                "n_iter": args.n_iter,
                "seed": args.seed,
            },
            f,
        )

    print("trained and saved:", args.out)
    print("weights:", getattr(clf, "w_", None))
    print("bias:", getattr(clf, "b_", None))
    print("updates per epoch:", getattr(clf, "errors_", []))


if __name__ == "__main__":
    main()
