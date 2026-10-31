import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from butterfly.features import Sites, featurize

PARAMS = {
    "objective": "cross_entropy",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
}


@dataclass
class WinModel:
    booster: lgb.Booster
    sites: Sites | None
    scale: float = 1.0

    def predict(self, states: np.ndarray, draw_possible: float = 0.0) -> np.ndarray:
        x, _ = featurize(states, self.sites, draw_possible)
        return self.booster.predict(x)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path / "model.txt"))
        (path / "meta.json").write_text(json.dumps({"sites": self.sites, "scale": self.scale}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "WinModel":
        meta = json.loads((path / "meta.json").read_text())
        sites = {k: tuple(v) for k, v in meta["sites"].items()} if meta["sites"] else None
        return cls(lgb.Booster(model_file=str(path / "model.txt")), sites, meta.get("scale", 1.0))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train(x: np.ndarray, y: np.ndarray, names: list[str], valid: np.ndarray, rounds: int = 2000,
          init_model: lgb.Booster | None = None, params: dict | None = None) -> tuple[lgb.Booster, dict]:
    params = {**PARAMS, **(params or {})}
    train_set = lgb.Dataset(x[~valid], y[~valid], feature_name=names, free_raw_data=False)
    valid_set = lgb.Dataset(x[valid], y[valid], reference=train_set)
    booster = lgb.train(
        params, train_set, num_boost_round=rounds, valid_sets=[valid_set],
        init_model=init_model, callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    p = booster.predict(x[valid], num_iteration=booster.best_iteration)
    base = np.full_like(p, y[~valid].mean())
    report = {
        "train_rows": int((~valid).sum()),
        "valid_rows": int(valid.sum()),
        "best_iteration": booster.best_iteration,
        "valid_log_loss": log_loss(y[valid], p),
        "valid_log_loss_constant": log_loss(y[valid], base),
        "valid_brier": float(np.mean((p - y[valid]) ** 2)),
    }
    return booster, report
