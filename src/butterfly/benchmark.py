from collections import defaultdict, deque

import numpy as np
import polars as pl

from butterfly.baselines import SwingModel, add_composites, baseline_ratings, component_norms
from butterfly.dataset import Match
from butterfly.model import WinModel, log_loss
from butterfly.ratings import rate_match
from butterfly.training import time_split

SYSTEMS = ["butterfly", "leetify_like", "hltv_like"]


def rating_table(matches: list[Match], model: WinModel, swing: SwingModel, n_perm: int = 8,
                 log=print) -> pl.DataFrame:
    tables, base_tables = [], []
    for n, m in enumerate(matches):
        ours, _ = rate_match(model, m, n_perm=n_perm)
        base = baseline_ratings(m, swing)
        teams = dict(m.roster.sort("round").select("steamid", "team").unique("steamid", keep="last").iter_rows())
        base = base.with_columns(pl.col("steamid").replace_strict(teams, default=0).alias("team"))
        tables.append(ours.select("steamid", "team", pl.col("rating").alias("butterfly"))
                      .with_columns(pl.lit(n).alias("match")))
        base_tables.append(base.with_columns(pl.lit(n).alias("match")))
        if log and (n + 1) % 25 == 0:
            log(f"rated {n + 1}/{len(matches)}")
    norms = component_norms(base_tables)
    base = add_composites(pl.concat(base_tables), norms).select("match", "steamid", "leetify_like", "hltv_like")
    return pl.concat(tables).join(base, on=["match", "steamid"], how="left")


def _fit_slope(d: np.ndarray, y: np.ndarray) -> float:
    w = 0.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-w * d))
        grad = np.sum((p - y) * d)
        hess = np.sum(p * (1 - p) * d * d) + 1e-9
        w -= grad / hess
    return w


def evaluate(matches: list[Match], table: pl.DataFrame, history: int = 10, valid_share: float = 0.15,
             min_known: int = 3) -> dict:
    valid = time_split(len(matches), valid_share)
    past: dict[str, dict[str, deque]] = {s: defaultdict(lambda: deque(maxlen=history)) for s in SYSTEMS}
    rows = []
    by_match = {k[0]: v for k, v in table.partition_by("match", as_dict=True).items()}
    for n, m in enumerate(matches):
        part = by_match.get(n)
        if part is None:
            continue
        winner = m.meta["winner"]
        y = 0.5 if winner == 0 else float(winner == 1)
        row = {"match": n, "valid": bool(valid[n]), "y": y}
        known = 0
        for system in SYSTEMS:
            strength = {}
            for team in (1, 2):
                sids = part.filter(pl.col("team") == team)["steamid"].to_list()
                hist = [np.mean(past[system][s]) for s in sids if past[system][s]]
                if system == SYSTEMS[0]:
                    known += len(hist)
                strength[team] = np.mean(hist) if hist else np.nan
            row[system] = strength[1] - strength[2]
        row["known"] = known
        rows.append(row)
        for system in SYSTEMS:
            for sid, value in part.select("steamid", system).iter_rows():
                if value is not None and not np.isnan(value):
                    past[system][sid].append(value)

    df = pl.DataFrame(rows).filter(pl.col("known") >= min_known)
    report = {"history": history, "evaluated": int(df["valid"].sum()), "fit_on": int((~df["valid"]).sum())}
    fit, test = df.filter(~pl.col("valid")), df.filter(pl.col("valid"))
    if test.is_empty() or fit.is_empty():
        report["note"] = "not enough matches with player history"
        return report
    y_test = test["y"].to_numpy()
    report["constant"] = {"log_loss": log_loss(y_test, np.full(len(y_test), 0.5)), "accuracy": 0.5}
    for system in SYSTEMS:
        d_fit = np.nan_to_num(fit[system].to_numpy())
        d_test = np.nan_to_num(test[system].to_numpy())
        w = _fit_slope(d_fit, fit["y"].to_numpy())
        p = 1 / (1 + np.exp(-w * d_test))
        decided = y_test != 0.5
        report[system] = {
            "log_loss": log_loss(y_test, p),
            "accuracy": float(np.mean((p[decided] > 0.5) == (y_test[decided] > 0.5))) if decided.any() else None,
            "slope": w,
        }
    report["stability"] = stability(table)
    return report


def stability(table: pl.DataFrame, min_matches: int = 10) -> dict:
    out = {}
    for system in SYSTEMS:
        halves = (table.sort("match").with_columns(pl.int_range(pl.len()).over("steamid").alias("k"))
                  .group_by("steamid").agg(
                      pl.len().alias("n"),
                      pl.col(system).filter(pl.col("k") % 2 == 0).mean().alias("even"),
                      pl.col(system).filter(pl.col("k") % 2 == 1).mean().alias("odd"))
                  .filter(pl.col("n") >= min_matches).drop_nulls())
        out[system] = {"players": halves.height,
                       "split_half_r": float(np.corrcoef(halves["even"], halves["odd"])[0, 1]) if halves.height > 2 else None}
    return out
