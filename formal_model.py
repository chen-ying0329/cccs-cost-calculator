from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn


class FeatureNet(nn.Module):
    def __init__(self, emb_size: int):
        super().__init__()
        self._layers = nn.ModuleList(
            [
                nn.Linear(1, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 64),
                nn.LeakyReLU(),
                nn.Linear(64, emb_size),
                nn.LeakyReLU(),
            ]
        )

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class HONAM(nn.Module):
    def __init__(self, n_features: int, order: int = 2, emb_size: int = 32):
        super().__init__()
        self._order = order
        self._emb_size = emb_size
        self._feature_nets = nn.ModuleList([FeatureNet(emb_size) for _ in range(n_features)])
        self._output_layer = nn.Linear(order * emb_size, 1)

    def forward(self, x):
        columns = x.T.unsqueeze(dim=2)
        encoded = [net(column) for net, column in zip(self._feature_nets, columns)]
        encoded = torch.stack(encoded, dim=1)
        powers = [1, encoded.sum(dim=1)]
        interactions = [1, encoded.sum(dim=1)]
        for order in range(2, self._order + 1):
            powers.append(encoded.pow(order).sum(dim=1))
            current = 0
            for j in range(1, order + 1):
                current += pow(-1, j + 1) * powers[j] * interactions[order - j]
            interactions.append(current / order)
        return self._output_layer(torch.concat(interactions[1:], dim=1))

    def contribution(self, x: np.ndarray, *feature_ids: int) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.tensor(x, dtype=torch.float32)
            columns = tensor.T.unsqueeze(dim=2)
            encoded = torch.stack(
                [net(column) for net, column in zip(self._feature_nets, columns)], dim=1
            )
            selected = encoded[:, feature_ids].prod(dim=1)
            weight = self._output_layer.weight.T[
                (len(feature_ids) - 1) * self._emb_size : len(feature_ids) * self._emb_size
            ]
            return (selected @ weight).cpu().numpy().reshape(-1)


class FrozenHONAMEnsemble:
    """Five-seed HONAM-M3 ensemble frozen on the development cohort."""

    def __init__(self, assets_dir: Path):
        self.assets_dir = Path(assets_dir)
        self.preprocessors = []
        self.models = []
        self.feature_names: list[str] | None = None
        for seed in range(5):
            preprocessor = joblib.load(self.assets_dir / f"preprocessor_seed_{seed}.joblib")
            checkpoint = torch.load(
                self.assets_dir / f"honam_seed_{seed}.pt",
                map_location="cpu",
                weights_only=True,
            )
            names = list(checkpoint["feature_names"])
            if self.feature_names is None:
                self.feature_names = names
            elif names != self.feature_names:
                raise RuntimeError("五个HONAM种子的特征顺序不一致")
            model = HONAM(
                n_features=len(names),
                emb_size=int(checkpoint["emb_size"]),
                order=int(checkpoint["order"]),
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self.preprocessors.append(preprocessor)
            self.models.append(model)

    def predict_proba(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        seed_probabilities = []
        for preprocessor, model in zip(self.preprocessors, self.models):
            transformed = preprocessor.transform(frame)
            with torch.no_grad():
                probability = torch.sigmoid(
                    model(torch.tensor(transformed, dtype=torch.float32))
                ).squeeze(1)
            seed_probabilities.append(probability.cpu().numpy())
        matrix = np.column_stack(seed_probabilities)
        return matrix.mean(axis=1), matrix

    def local_main_effects(self, frame: pd.DataFrame) -> list[dict]:
        effects = np.zeros((len(frame), len(self.feature_names)), dtype=float)
        for preprocessor, model in zip(self.preprocessors, self.models):
            transformed = preprocessor.transform(frame)
            for feature_id in range(len(self.feature_names)):
                effects[:, feature_id] += model.contribution(transformed, feature_id) / len(self.models)
        first = effects[0]
        order = np.argsort(np.abs(first))[::-1][:6]
        return [
            {
                "label": self.feature_names[index],
                "value": float(first[index]),
                "direction": "up" if first[index] >= 0 else "down",
            }
            for index in order
        ]
