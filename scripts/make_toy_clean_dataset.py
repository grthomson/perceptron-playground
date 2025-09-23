from pathlib import Path

import pandas as pd
from faker import Faker

OUT = Path(__file__).resolve().parents[1] / "data" / "toy_people_clean.csv"


def main(n_rows: int = 100, seed: int = 42):
    fake = Faker("en_GB")  # UK-style addresses & postcodes
    Faker.seed(seed)

    rows = []
    for i in range(1, n_rows + 1):
        rows.append(
            {
                "id": i,
                "forename": fake.first_name(),
                "surname": fake.last_name(),
                "address": fake.street_address(),
                "city": fake.city(),
                "postcode": fake.postcode(),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT} with {n_rows} rows")


if __name__ == "__main__":
    main()
