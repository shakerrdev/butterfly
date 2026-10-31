import numpy as np
import polars as pl

from butterfly.dataset import Match, site_centers
from butterfly.features import featurize
from butterfly.model import WinModel, train


def training_rows(match: Match, stride: int) -> np.ndarray:
    kinds = match.info["kind"].to_numpy()
    live_pos = np.cumsum(kinds == "live")
    return np.where((kinds != "live") | (live_pos % stride == 0))[0]


def build(matches: list[Match], sites, stride: int, target: str = "match") -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    xs, ys, owners = [], [], []
    names: list[str] = []
    for n, m in enumerate(matches):
        rows = training_rows(m, stride)
        x, names = featurize(m.states[rows], sites, 1.0 if m.meta.get("draw_possible") else 0.0)
        xs.append(x)
        ys.append(m.labels(target)[rows])
        owners.append(np.full(len(rows), n))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(owners), names


def time_split(n_matches: int, valid_share: float) -> np.ndarray:
    n_valid = max(1, int(round(n_matches * valid_share))) if n_matches > 1 else 0
    valid = np.zeros(n_matches, dtype=bool)
    if n_valid:
        valid[-n_valid:] = True
    return valid


def fit(matches: list[Match], stride: int = 4, valid_share: float = 0.15, base: WinModel | None = None,
        rounds: int = 2000, params: dict | None = None, target: str = "match") -> tuple[WinModel, dict, np.ndarray]:
    sites = base.sites if base else site_centers(matches)
    x, y, owner, names = build(matches, sites, stride, target)
    valid_matches = time_split(len(matches), valid_share)
    valid = valid_matches[owner] if valid_matches.any() else np.zeros(len(y), dtype=bool)
    if not valid.any():
        valid = np.arange(len(y)) % 5 == 0
    booster, report = train(x, y, names, valid, rounds=rounds,
                            init_model=base.booster if base else None, params=params)
    report["matches"] = len(matches)
    report["valid_matches"] = int(valid_matches.sum())
    report["sites"] = sites
    report["target"] = target
    return WinModel(booster, sites, base.scale if base else 1.0), report, valid_matches


def calibrate_scale(model: WinModel, matches: list[Match], target_sd: float = 0.25, n_perm: int = 8) -> float:
    from butterfly.ratings import rate_match

    per_round = []
    for m in matches:
        table, _ = rate_match(model, m, n_perm=n_perm)
        table = table.filter(pl.col("rounds") >= 10)
        per_round.extend((table["wpa"] / table["rounds"]).to_list())
    if len(per_round) < 2:
        return model.scale
    return float(np.std(per_round) / target_sd)
