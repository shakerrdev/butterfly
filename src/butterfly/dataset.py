import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from butterfly.features import Sites
from butterfly.state import GLOBAL_COLS, PLAYER_COLS, SLOTS


@dataclass
class Match:
    path: Path
    meta: dict
    info: pl.DataFrame
    states: np.ndarray
    roster: pl.DataFrame
    events: pl.DataFrame

    @property
    def name(self) -> str:
        return self.path.name

    def ct_team(self) -> np.ndarray:
        teams = {r: t for r, t in self.roster.filter(pl.col("slot") < 5).unique("round")
                 .select("round", "team").iter_rows()}
        return np.array([teams.get(r, 1) for r in self.info["round"].to_list()])

    def round_labels(self) -> np.ndarray:
        winners = {r: w for r, w in self.info.with_row_index("i").filter(pl.col("kind") == "end")
                   .select("round", "i").iter_rows()}
        ct_won = {r: self.states[i, GLOBAL_COLS.index("round_winner")] > 0 for r, i in winners.items()}
        return np.array([float(ct_won.get(r, 0.5)) for r in self.info["round"].to_list()])

    def labels(self, target: str = "match") -> np.ndarray:
        if target == "round":
            return self.round_labels()
        winner = self.meta["winner"]
        if winner == 0:
            return np.full(len(self.states), 0.5)
        return (self.ct_team() == winner).astype(np.float64)


STATE_NAMES = list(GLOBAL_COLS) + [f"s{slot}_{c}" for slot in range(SLOTS) for c in PLAYER_COLS]


def load_match(path: Path) -> Match:
    meta = json.loads((path / "match.json").read_text())
    states = pl.read_parquet(path / "states.parquet")
    info = states.select("round", "kind", "tick")
    mat = states.select(STATE_NAMES).to_numpy().astype(np.float64)
    return Match(path, meta, info, mat, pl.read_parquet(path / "roster.parquet"),
                 pl.read_parquet(path / "events.parquet"))


def match_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob("match.json"))


def load_matches(root: Path, map_name: str | None = None, complete_only: bool = True) -> list[Match]:
    out = []
    for d in match_dirs(root):
        meta = json.loads((d / "match.json").read_text())
        if map_name and meta.get("map") != map_name:
            continue
        if complete_only and not meta.get("complete"):
            continue
        out.append(load_match(d))
    return sorted(out, key=lambda m: str(m.meta.get("order_key", "")))


def site_centers(matches: list[Match]) -> Sites | None:
    plants = [m.events.filter((pl.col("kind") == "plant") & pl.col("site").is_not_null() & pl.col("x").is_not_null())
              for m in matches]
    plants = [p for p in plants if not p.is_empty()]
    if not plants:
        return None
    df = pl.concat(plants).group_by("site").agg(pl.col("x").median(), pl.col("y").median()).sort("site")
    names = {1: "A", 2: "B"}
    return {names.get(s, str(s)): (float(x), float(y)) for s, x, y in df.iter_rows()}
