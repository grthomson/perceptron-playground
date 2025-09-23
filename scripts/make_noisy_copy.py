import argparse
from pathlib import Path

import pandas as pd

from data_prep.noise_imputer import NoiseImputer

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main():
    ap = argparse.ArgumentParser(
        description="Create noisy copy + positive labels from clean toy dataset"
    )
    ap.add_argument("--input", default=str(DATA_DIR / "toy_people_clean.csv"))
    ap.add_argument("--out-noisy", default=str(DATA_DIR / "toy_people_noisy.csv"))
    ap.add_argument("--out-labels", default=str(DATA_DIR / "toy_labels.csv"))

    # error rates
    ap.add_argument("--typo-forename", type=float, default=0.20)
    ap.add_argument("--typo-surname", type=float, default=0.20)
    ap.add_argument("--addr-abbrev", type=float, default=0.30)
    ap.add_argument("--postcode-squash", type=float, default=0.60)
    ap.add_argument("--random-casing", type=float, default=0.15)
    ap.add_argument("--swap-name-fields", type=float, default=0.05)

    # reproducibility
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    clean = pd.read_csv(args.input)

    imputer = NoiseImputer(
        typo_rate_forename=args.typo_forename,
        typo_rate_surname=args.typo_surname,
        addr_abbrev_rate=args.addr_abbrev,
        postcode_squash_rate=args.postcode_squash,
        random_casing_rate=args.random_casing,
        swap_name_fields_rate=args.swap_name_fields,
        seed=args.seed,
    )

    noisy = imputer.make_noisy(clean)
    labels = imputer.make_positive_labels(clean, noisy)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    noisy.to_csv(args.out_noisy, index=False)
    labels.to_csv(args.out_labels, index=False)

    print(f"wrote {args.out_noisy}")
    print(f"wrote {args.out_labels}  (rows={len(labels)})")


if __name__ == "__main__":
    main()
