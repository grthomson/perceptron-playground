import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

_ABBREV = {
    " street": " st",
    " road": " rd",
    " avenue": " ave",
    " square": " sq",
    " lane": " ln",
    " way": " wy",
}
_WS_DASH = re.compile(r"[ \t-]+")


@dataclass
class NoiseImputer:
    """Create a noisy copy of a 'people' dataset with parameterised error rates.

    Expected columns on input: id, forename, surname, address, city, postcode
    """

    typo_rate_forename: float = 0.2
    typo_rate_surname: float = 0.2
    addr_abbrev_rate: float = 0.3
    postcode_squash_rate: float = 0.6
    random_casing_rate: float = 0.15
    swap_name_fields_rate: float = 0.05
    seed: int | None = 42

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    # -------- atomic noise ops --------

    def _maybe(self, p: float) -> bool:
        return bool(self.rng.random() < p)

    def _typo(self, s: str) -> str:
        if not s:
            return s
        i = int(self.rng.integers(0, len(s)))
        if self.rng.random() < 0.5:
            # delete one character
            return s[:i] + s[i + 1 :]
        # swap adjacent characters if possible
        if i < len(s) - 1:
            return s[:i] + s[i + 1] + s[i] + s[i + 2 :]
        return s

    def _abbr_address(self, addr: str) -> str:
        a = f" {addr.strip().lower()} "
        for k, v in _ABBREV.items():
            a = a.replace(k, v)
        return a.strip()

    def _squash_postcode(self, pc: str) -> str:
        return _WS_DASH.sub("", str(pc))

    def _random_case(self, s: str) -> str:
        return s.lower() if self.rng.random() < 0.5 else s.upper()

    # -------- high-level API --------

    def make_noisy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a noisy copy; keeps same row order for easy 1:1 labels."""
        required = {"id", "forename", "surname", "address", "city", "postcode"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        noisy = df.copy(deep=True)

        # forename / surname typos
        noisy["forename"] = [
            self._typo(x) if self._maybe(self.typo_rate_forename) else x
            for x in noisy["forename"].astype(str)
        ]
        noisy["surname"] = [
            self._typo(x) if self._maybe(self.typo_rate_surname) else x
            for x in noisy["surname"].astype(str)
        ]

        # occasionally swap forename <-> surname
        if self.swap_name_fields_rate > 0:
            swap_mask = self.rng.random(len(noisy)) < self.swap_name_fields_rate
            f_tmp = noisy.loc[swap_mask, "forename"].copy()
            noisy.loc[swap_mask, "forename"] = noisy.loc[swap_mask, "surname"]
            noisy.loc[swap_mask, "surname"] = f_tmp

        # address abbreviations
        noisy["address"] = [
            self._abbr_address(a) if self._maybe(self.addr_abbrev_rate) else a
            for a in noisy["address"].astype(str)
        ]

        # postcode squashing (remove spaces/dashes)
        noisy["postcode"] = [
            self._squash_postcode(p) if self._maybe(self.postcode_squash_rate) else p
            for p in noisy["postcode"].astype(str)
        ]

        # random casing on names
        if self.random_casing_rate > 0:
            mask = self.rng.random(len(noisy)) < self.random_casing_rate
            noisy.loc[mask, "forename"] = noisy.loc[mask, "forename"].map(
                self._random_case
            )
            noisy.loc[mask, "surname"] = noisy.loc[mask, "surname"].map(
                self._random_case
            )

        # keep right-table id distinct but aligned by row
        noisy = noisy.rename(columns={"id": "id_right"})
        return noisy

    def make_positive_labels(
        self, left: pd.DataFrame, right: pd.DataFrame
    ) -> pd.DataFrame:
        """Return a labels DataFrame with 1:1 positives based on row alignment."""
        if "id" not in left.columns or "id_right" not in right.columns:
            raise ValueError("left requires 'id'; right requires 'id_right'")
        return pd.DataFrame(
            {
                "left_id": left["id"].to_numpy(),
                "right_id": right["id_right"].to_numpy(),
                "label": 1,
            }
        )
