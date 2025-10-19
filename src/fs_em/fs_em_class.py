# src/fs_em/fs_em_class.py
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


class FellegiSunterEM:
    """Fellegi–Sunter record linkage with EM."""

    def __init__(
        self,
        max_iter: int = 50,
        tol: float = 1e-4,
        clip: float = 1e-6,
        prior: float = 0.5,
        zero_weight_for_missing: bool = True,
        verbose: bool = False,
        random_state: int | None = None,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.clip = clip
        self.prior = prior
        self.zero_weight_for_missing = zero_weight_for_missing
        self.verbose = verbose
        self.random_state = random_state

        # populated after fit
        self.variables_: list[str] = []
        self.params_: pd.DataFrame | None = None
        self.weights_: pd.DataFrame | None = None
        self.prior_: float | None = None
        self.delta_history_: list[float] = []

    # ---------------- public API ---------------- #

    def fit(
        self, comparisons: pd.DataFrame, variables: Sequence[str]
    ) -> FellegiSunterEM:
        """Estimate m/u parameters, weights, and prior from {-1,0,1} comparisons."""
        self.variables_ = list(variables)
        mat = comparisons[self.variables_].to_numpy(int)

        agree = mat == 1
        disagree = mat == 0
        missing = mat == -1

        rng = np.random.default_rng(self.random_state)
        m_agree = self._init_vec(0.8, mat.shape[1], rng)
        m_disagree = self._init_vec(0.1, mat.shape[1], rng)
        m_missing = self._init_vec(0.1, mat.shape[1], rng)
        u_agree = self._init_vec(0.2, mat.shape[1], rng)
        u_disagree = self._init_vec(0.7, mat.shape[1], rng)
        u_missing = self._init_vec(0.1, mat.shape[1], rng)

        p = float(self.prior)
        self.delta_history_ = []

        for it in range(self.max_iter):
            # --- E-step ---
            m_prob = (
                m_agree[None, :] * agree
                + m_disagree[None, :] * disagree
                + m_missing[None, :] * missing
            )
            u_prob = (
                u_agree[None, :] * agree
                + u_disagree[None, :] * disagree
                + u_missing[None, :] * missing
            )

            m_like = np.prod(m_prob, axis=1)
            u_like = np.prod(u_prob, axis=1)
            g = (p * m_like) / (p * m_like + (1.0 - p) * u_like)

            # --- M-step ---
            match_mass = g.sum()
            non_match_mass = (1.0 - g).sum()

            m_agree_new = (g[:, None] * agree).sum(axis=0) / match_mass
            m_disagree_new = (g[:, None] * disagree).sum(axis=0) / match_mass
            m_missing_new = (g[:, None] * missing).sum(axis=0) / match_mass

            u_agree_new = ((1.0 - g)[:, None] * agree).sum(axis=0) / non_match_mass
            u_disagree_new = ((1.0 - g)[:, None] * disagree).sum(
                axis=0
            ) / non_match_mass
            u_missing_new = ((1.0 - g)[:, None] * missing).sum(axis=0) / non_match_mass

            p_new = float(g.mean())

            for arr in (
                m_agree_new,
                m_disagree_new,
                m_missing_new,
                u_agree_new,
                u_disagree_new,
                u_missing_new,
            ):
                np.clip(arr, self.clip, 1.0 - self.clip, out=arr)

            delta = max(
                np.abs(m_agree_new - m_agree).max(),
                np.abs(m_disagree_new - m_disagree).max(),
                np.abs(m_missing_new - m_missing).max(),
                np.abs(u_agree_new - u_agree).max(),
                np.abs(u_disagree_new - u_disagree).max(),
                np.abs(u_missing_new - u_missing).max(),
                abs(p_new - p),
            )
            self.delta_history_.append(float(delta))
            if self.verbose:
                print(f"iter={it+1:02d}  p={p_new:.6f}  Δ={delta:.2e}")

            m_agree, m_disagree, m_missing = m_agree_new, m_disagree_new, m_missing_new
            u_agree, u_disagree, u_missing = u_agree_new, u_disagree_new, u_missing_new
            p = p_new
            if delta < self.tol:
                break

        self.params_ = pd.DataFrame(
            {
                "variable": self.variables_,
                "m_agree": m_agree,
                "m_disagree": m_disagree,
                "m_missing": m_missing,
                "u_agree": u_agree,
                "u_disagree": u_disagree,
                "u_missing": u_missing,
            }
        )
        self.prior_ = p
        self.weights_ = self._compute_weights(self.params_)
        return self

    def score(self, comparisons: pd.DataFrame) -> pd.Series:
        """Return FS log2-likelihood ratio scores for each pair."""
        self._check_fitted()
        gamma = comparisons[self.variables_]
        w = self.weights_.set_index("variable")

        scores = pd.Series(0.0, index=gamma.index)
        for v in self.variables_:
            vals = gamma[v].astype(int)
            w_row = w.loc[v]
            w_missing = (
                0.0 if self.zero_weight_for_missing else float(w_row["w_missing"])
            )
            inc = np.select(
                [vals == 1, vals == 0, vals == -1],
                [float(w_row["w_agree"]), float(w_row["w_disagree"]), w_missing],
                default=0.0,
            )
            scores += inc
        return scores

    def predict_proba(
        self, comparisons: pd.DataFrame, prior: float | None = None
    ) -> pd.Series:
        """Return posterior P(match | comparisons)."""
        self._check_fitted()
        prior = float(self.prior_ if prior is None else prior)
        k = (1.0 - prior) / prior
        s = self.score(comparisons).to_numpy(float)
        return pd.Series(1.0 / (1.0 + k * (2.0 ** (-s))), index=comparisons.index)

    # ---------------- helpers ---------------- #

    @staticmethod
    def build_comparisons(df: pd.DataFrame, variables: Sequence[str]) -> pd.DataFrame:
        """Create {-1,0,1} comparison columns from raw candidate pairs."""
        out = pd.DataFrame(index=df.index)
        out[["record_id_df1", "record_id_df2"]] = df[["record_id_df1", "record_id_df2"]]
        for v in variables:
            out[v] = FellegiSunterEM._encode(df[f"{v}_df1"], df[f"{v}_df2"])
        return out

    @staticmethod
    def _encode(c1: pd.Series, c2: pd.Series) -> pd.Series:
        missing = c1.isna() | c2.isna()
        agree = (c1 == c2).astype("Int64")
        return agree.where(~missing, -1).astype(int)

    @staticmethod
    def _init_vec(center: float, length: int, rng: np.random.Generator) -> np.ndarray:
        vec = np.full(length, float(center)) + 0.01 * (rng.random(length) - 0.5)
        return np.clip(vec, 1e-6, 1 - 1e-6)

    @staticmethod
    def _compute_weights(params: pd.DataFrame) -> pd.DataFrame:
        out = params.copy()
        out["w_agree"] = np.log2(out["m_agree"] / out["u_agree"])
        out["w_disagree"] = np.log2(out["m_disagree"] / out["u_disagree"])
        out["w_missing"] = np.log2(out["m_missing"] / out["u_missing"])
        return out[["variable", "w_agree", "w_disagree", "w_missing"]]

    def _check_fitted(self) -> None:
        if self.params_ is None or self.weights_ is None or self.prior_ is None:
            raise RuntimeError("Call .fit(...) before scoring or predicting.")
