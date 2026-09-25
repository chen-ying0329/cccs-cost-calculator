"""Compatibility classes used by the frozen HONAM preprocessors.

The training code serialized these classes under this module name.  Keep the
class names and attributes stable so joblib can load the five frozen objects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer


class BaseImputer:
    def __init__(self, categorical: list[str], continuous: list[str], binary: list[str]):
        self.categorical = categorical
        self.continuous = continuous
        self.binary = binary
        self.fill_values: dict[str, Any] = {}

    def fit(self, frame: pd.DataFrame) -> "BaseImputer":
        for column in self.categorical + self.binary:
            mode = frame[column].dropna().mode()
            self.fill_values[column] = mode.iloc[0] if len(mode) else 0
        for column in self.continuous:
            values = pd.to_numeric(frame[column], errors="coerce")
            self.fill_values[column] = float(values.median())
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for column in self.categorical:
            out[column] = out[column].fillna(self.fill_values[column]).astype(str)
        for column in self.binary:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(
                self.fill_values[column]
            )
        for column in self.continuous:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(
                self.fill_values[column]
            )
        if out.isna().any().any():
            raise RuntimeError("填补后仍存在缺失值")
        return out


class HONAMPaperPreprocessor:
    def __init__(
        self,
        feature_names: list[str],
        categorical: list[str],
        continuous: list[str],
        binary: list[str],
        seed: int,
    ):
        self.feature_names = feature_names
        self.categorical = categorical
        self.continuous = continuous
        self.binary = binary
        self.seed = seed
        self.imputer = BaseImputer(categorical, continuous, binary)
        self.category_maps: dict[str, dict[str, int]] = {}
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.quantile: QuantileTransformer | None = None

    def _numeric_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        cleaned = self.imputer.transform(frame[self.feature_names])
        matrix = np.zeros((len(cleaned), len(self.feature_names)), dtype=np.float64)
        for index, column in enumerate(self.feature_names):
            if column in self.categorical:
                mapping = self.category_maps[column]
                matrix[:, index] = cleaned[column].map(mapping).fillna(-1).astype(float)
            else:
                matrix[:, index] = pd.to_numeric(cleaned[column], errors="coerce").to_numpy()
            if column in self.continuous:
                matrix[:, index] = (matrix[:, index] - self.means[column]) / self.stds[column]
        return matrix

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.quantile is None:
            raise RuntimeError("HONAM预处理器尚未拟合")
        matrix = self._numeric_matrix(frame[self.feature_names])
        return self.quantile.transform(matrix).astype(np.float32)
