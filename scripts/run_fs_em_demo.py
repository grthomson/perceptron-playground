from pathlib import Path

import pandas as pd

from fs_em.fs_em_class import FellegiSunterEM


def main():
    DATA = Path(__file__).resolve().parents[1] / "data" / "candidate_pairs.csv"
    df = pd.read_csv(DATA)

    variables = ["forename", "surname", "year_of_birth", "sex", "postcode"]
    comps = FellegiSunterEM.build_comparisons(df, variables)

    model = FellegiSunterEM(max_iter=50, tol=1e-4, verbose=True, random_state=1)
    model.fit(comps, variables)

    probs = model.predict_proba(comps)
    print(df.assign(match_probability=probs))


if __name__ == "__main__":
    main()
