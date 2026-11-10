"""Phase 1 features -> network inputs (integer category indices + scaled numerics).

Every statistic (category vocabulary, rare-bucket cutoff, imputation medians,
scaler) is fitted on the rows a model is fitted on, and only on those: per CV
fold on that fold's training rows, and for the final model on the training
rows available at ``config.FINAL_FIT_DATE``. This mirrors Phase 1, where the
sklearn pipelines refit their encoder and scaler inside every fold. Fitting
goes through ``fit_encoder``, which enforces the leakage guards.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from aerocast import config as p1
from aerocast import data as p1_data
from aerocast.pipeline import Prepared

from . import config, guards

UNK = 0  # shared index for rare, unseen and missing categories


# ---------------------------------------------------------------------------
# Row bookkeeping
# ---------------------------------------------------------------------------
def row_info(P: Prepared) -> pd.DataFrame:
    """Split, label, date and label-availability date for every modelling row,
    in the row order of ``P.meta`` / ``P.X``."""
    n = len(P.meta)
    split = np.empty(n, dtype=object)
    y = np.full(n, -1, dtype=int)
    avail = np.empty(n, dtype="datetime64[ns]")
    for s, i in P.idx.items():
        split[i], y[i], avail[i] = s, P.labels.y[s], P.labels.avail[s]
    return pd.DataFrame({"split": split, "y": y, "date": P.meta["date"].to_numpy(),
                         "month": P.meta["month"].to_numpy(), "avail": avail})


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def id_columns(meta: pd.DataFrame, ids=config.ID_COLS) -> pd.DataFrame:
    """High-cardinality identifiers built from schedule fields only.

    Uses carrier, flight, origin, hour and tailnum, all known at the decision
    clock; never a post-departure column (``aerocast.data.POST_DEPARTURE_COLS``).
    """
    build = {
        "tailnum": lambda m: m["tailnum"],
        "carrier_flight": lambda m: m["carrier"] + "_" + m["flight"].astype(str),
        "origin_hour": lambda m: m["origin"] + "_" + m["hour"].astype(str),
    }
    unknown = set(ids) - set(build)
    if unknown:
        raise ValueError(f"unknown id columns: {sorted(unknown)}")
    return pd.DataFrame({c: build[c](meta) for c in ids}, index=meta.index)


def model_inputs(P: Prepared, ids=(), num_cols=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(categorical frame, numeric frame) for all modelling rows.

    Categorical = the Phase 1 categorical features plus optional ``ids``;
    numeric = the Phase 1 numeric features (or the subset ``num_cols``, for
    ablations), in Phase 1 column order.
    """
    cats = P.X[P.cat_cols].astype(object)
    if ids:
        cats = pd.concat([cats, id_columns(P.meta, ids)], axis=1)
    return cats, P.X[list(num_cols) if num_cols is not None else P.num_cols]


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
class CategoryVocab:
    """value -> index; values with fewer than ``min_freq`` fitting rows, unseen
    values and missing values all map to ``UNK`` (0)."""

    def __init__(self, min_freq: int = config.MIN_CAT_FREQ):
        self.min_freq = min_freq

    def fit(self, values: pd.Series) -> "CategoryVocab":
        counts = pd.Series(values).dropna().value_counts()
        kept = sorted(counts.index[counts >= self.min_freq].astype(str))
        self.index_ = {v: k + 1 for k, v in enumerate(kept)}
        return self

    @property
    def cardinality(self) -> int:
        """Embedding-table size, including the UNK row."""
        return len(self.index_) + 1

    def transform(self, values: pd.Series) -> np.ndarray:
        s = pd.Series(values)
        out = s.astype(str).map(self.index_).where(s.notna())
        return out.fillna(UNK).astype(np.int64).to_numpy()


@dataclass
class Encoded:
    cat: np.ndarray              # int64 [n, n_cat]
    num: np.ndarray              # float32 [n, n_num_out]
    cat_cols: list[str]
    num_cols: list[str]          # output names, incl. missing indicators
    cardinalities: list[int]

    def __len__(self) -> int:
        return len(self.cat)

    def dataset(self, y) -> TensorDataset:
        return TensorDataset(torch.from_numpy(self.cat), torch.from_numpy(self.num),
                             torch.from_numpy(np.array(y, dtype=np.float32)))


class TabularEncoder:
    """Per-column vocabularies + Phase 1's numeric preprocessing (median
    imputation with missing indicators, then standardisation)."""

    def __init__(self, min_freq: int = config.MIN_CAT_FREQ):
        self.min_freq = min_freq

    def fit(self, cats: pd.DataFrame, nums: pd.DataFrame) -> "TabularEncoder":
        self.cat_cols_ = list(cats.columns)
        self.vocabs_ = {c: CategoryVocab(self.min_freq).fit(cats[c]) for c in self.cat_cols_}
        self.num_ = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]).set_output(transform="default").fit(nums)
        self.num_cols_ = list(self.num_.get_feature_names_out(nums.columns))
        return self

    def transform(self, cats: pd.DataFrame, nums: pd.DataFrame) -> Encoded:
        cat = np.stack([self.vocabs_[c].transform(cats[c]) for c in self.cat_cols_], axis=1)
        num = self.num_.transform(nums).astype(np.float32)
        return Encoded(cat, num, self.cat_cols_, self.num_cols_,
                       [self.vocabs_[c].cardinality for c in self.cat_cols_])


def fit_encoder(P: Prepared, rows: np.ndarray, fit_date, ids=(),
                min_freq: int = config.MIN_CAT_FREQ, num_cols=None) -> TabularEncoder:
    """Fit the encoder on modelling-row positions ``rows`` for a model fitted
    at the decision clock of ``fit_date``.

    Guards: the rows must be training rows (G3/G4) whose labels are available
    by ``fit_date`` (G2), i.e. exactly the rows the network itself may use.
    """
    rows = np.asarray(rows)
    info = row_info(P).iloc[rows]
    guards.assert_rows_in_splits(info["split"], {"train"}, "encoder fit")
    guards.assert_labels_available(info["avail"], fit_date, "encoder fit")
    cats, nums = model_inputs(P, ids, num_cols)
    return TabularEncoder(min_freq).fit(cats.iloc[rows], nums.iloc[rows])


def final_fit_rows(P: Prepared) -> np.ndarray:
    """Training rows the final model may use: labels known for 2013-09-01."""
    info = row_info(P)
    ok = (info["split"].eq("train").to_numpy()
          & p1_data.outcomes_available_by(info["avail"], config.FINAL_FIT_DATE))
    return np.flatnonzero(ok)


# ---------------------------------------------------------------------------
# Cold start (descriptive; uses category membership only, no labels)
# ---------------------------------------------------------------------------
def cold_start_table(P: Prepared, ids=config.ID_COLS,
                     min_freq: int = config.MIN_CAT_FREQ) -> pd.DataFrame:
    """Share of rows per split whose category falls in the UNK bucket of a
    vocabulary fitted on the final training rows (rare or unseen in train)."""
    rows = final_fit_rows(P)
    cats, _ = model_inputs(P, ids)
    split = row_info(P)["split"]
    out = []
    for c in cats.columns:
        vocab = CategoryVocab(min_freq).fit(cats[c].iloc[rows])
        unk = vocab.transform(cats[c]) == UNK
        seen = cats[c].iloc[rows].dropna().astype(str).unique()
        unseen = ~cats[c].astype(str).isin(seen).to_numpy()
        for s in p1.SPLITS:
            m = split.eq(s).to_numpy()
            out.append({"column": c, "split": s, "train_categories": len(seen),
                        "embedded_categories": vocab.cardinality - 1,
                        "unk_share": float(unk[m].mean()),
                        "unseen_share": float(unseen[m].mean())})
    return pd.DataFrame(out)
