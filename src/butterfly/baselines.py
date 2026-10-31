from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from butterfly.dataset import Match
from butterfly.features import featurize

TRADE_WINDOW = 5 * 64
KILL_SHARES = {"killer": 0.35, "damage": 0.30, "flash": 0.15, "trade": 0.20}
SWING_FEATURES = ["ct_alive", "t_alive", "bomb_planted", "ct_buy", "t_buy"]


def buy_group(equip: np.ndarray) -> np.ndarray:
    return np.digitize(equip, [5000, 10000, 20000]).astype(np.float64)


@dataclass
class SwingModel:
    booster: lgb.Booster

    def predict(self, ct_alive, t_alive, planted, ct_buy, t_buy) -> np.ndarray:
        x = np.column_stack([ct_alive, t_alive, planted, ct_buy, t_buy]).astype(np.float32)
        return self.booster.predict(x)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path / "swing.txt"))

    @classmethod
    def load(cls, path: Path) -> "SwingModel":
        return cls(lgb.Booster(model_file=str(path / "swing.txt")))


def _round_start_buys(match: Match, names: list[str], x: np.ndarray) -> dict[int, tuple[float, float]]:
    info = match.info.with_row_index("i").filter(pl.col("kind") == "live").group_by("round").agg(pl.col("i").min())
    ct_i, t_i = names.index("ct_equip"), names.index("t_equip")
    return {r: (x[i, ct_i], x[i, t_i]) for r, i in info.iter_rows()}


def train_swing(matches: list[Match]) -> SwingModel:
    xs, ys = [], []
    for m in matches:
        x, names = featurize(m.states)
        buys = _round_start_buys(m, names, x)
        rounds = m.info["round"].to_numpy()
        live = (m.info["kind"] == "live").to_numpy()
        ct_buy = buy_group(np.array([buys.get(r, (0, 0))[0] for r in rounds]))
        t_buy = buy_group(np.array([buys.get(r, (0, 0))[1] for r in rounds]))
        cols = [x[:, names.index("ct_alive")], x[:, names.index("t_alive")], x[:, names.index("bomb_planted")],
                ct_buy, t_buy]
        xs.append(np.column_stack(cols)[live])
        ys.append(m.round_labels()[live])
    x, y = np.concatenate(xs).astype(np.float32), np.concatenate(ys)
    params = {"objective": "cross_entropy", "learning_rate": 0.1, "num_leaves": 31, "min_data_in_leaf": 50,
              "verbose": -1}
    return SwingModel(lgb.train(params, lgb.Dataset(x, y, feature_name=SWING_FEATURES), num_boost_round=200))


def baseline_ratings(match: Match, swing: SwingModel) -> pl.DataFrame:
    x, names = featurize(match.states)
    buys = _round_start_buys(match, names, x)
    stats = defaultdict(lambda: defaultdict(float))
    rounds_played = defaultdict(int)
    names_of = dict(match.roster.select("steamid", "name").unique("steamid").iter_rows())

    for rnd in sorted(set(match.roster["round"].to_list())):
        roster = match.roster.filter(pl.col("round") == rnd)
        side = {sid: ("ct" if slot < 5 else "t") for sid, slot in roster.select("steamid", "slot").iter_rows()}
        for sid in side:
            rounds_played[sid] += 1
        ev = match.events.filter(pl.col("round") == rnd).sort("tick")
        plant_tick = ev.filter(pl.col("kind") == "plant")["tick"].min()
        alive = {"ct": sum(1 for s in side.values() if s == "ct"), "t": sum(1 for s in side.values() if s == "t")}
        ct_buy, t_buy = (buy_group(np.array(v)) for v in buys.get(rnd, (0.0, 0.0)))
        damage_to = defaultdict(lambda: defaultdict(float))
        kills_in_round = defaultdict(int)
        recent_deaths: list[tuple[int, str, str | None]] = []
        dead, kast = set(), set()
        for e in ev.iter_rows(named=True):
            att, vic = e["attacker"], e["victim"]
            if e["kind"] == "damage" and att in side and vic in side and side[att] != side[vic]:
                damage_to[vic][att] += e["amount"]
                stats[att]["damage"] += min(e["amount"], 100.0)
            if e["kind"] != "death" or vic not in side:
                continue
            planted = float(plant_tick is not None and e["tick"] >= plant_tick)
            before = swing.predict(alive["ct"], alive["t"], planted, ct_buy, t_buy)[0]
            alive[side[vic]] -= 1
            after = swing.predict(alive["ct"], alive["t"], planted, ct_buy, t_buy)[0]
            delta = abs(after - before)
            dead.add(vic)
            stats[vic]["deaths"] += 1
            stats[vic]["swing"] -= delta
            enemy_kill = att in side and side[att] != side[vic]
            shares = {}
            if enemy_kill:
                shares[att] = shares.get(att, 0) + KILL_SHARES["killer"]
                stats[att]["kills"] += 1
                kills_in_round[att] += 1
                kast.add(att)
                dmg = {a: d for a, d in damage_to[vic].items() if a != att and d > 0}
                total = sum(dmg.values())
                for a, d in dmg.items():
                    shares[a] = shares.get(a, 0) + KILL_SHARES["damage"] * d / total
                    if d >= 40:
                        kast.add(a)
                if e["assister"] in side and e["flash_assist"] and side[e["assister"]] == side[att]:
                    shares[e["assister"]] = shares.get(e["assister"], 0) + KILL_SHARES["flash"]
                    kast.add(e["assister"])
                for tick, dead_sid, killer in recent_deaths:
                    if killer == vic and e["tick"] - tick <= TRADE_WINDOW and side.get(dead_sid) == side[att]:
                        shares[dead_sid] = shares.get(dead_sid, 0) + KILL_SHARES["trade"]
                        kast.add(dead_sid)
                        break
                norm = sum(shares.values())
                for sid, share in shares.items():
                    stats[sid]["swing"] += delta * share / norm
            recent_deaths.append((e["tick"], vic, att if enemy_kill else None))
        for sid in side:
            if sid not in dead:
                kast.add(sid)
                stats[sid]["survived"] += 1
            if sid in kast:
                stats[sid]["kast"] += 1
            stats[sid]["multi"] += float(kills_in_round[sid] >= 2)

    rows = []
    for sid, n in rounds_played.items():
        s = stats[sid]
        rows.append({
            "steamid": sid, "name": names_of.get(sid, ""), "rounds": n,
            "kpr": s["kills"] / n, "dpr": s["deaths"] / n, "adr": s["damage"] / n, "survival": s["survived"] / n,
            "kast": s["kast"] / n, "multi": s["multi"] / n, "swing": s["swing"] / n,
        })
    return pl.DataFrame(rows)


def add_composites(table: pl.DataFrame, norms: dict[str, float]) -> pl.DataFrame:
    leetify = pl.col("swing") * 100
    parts = [pl.col(c) / norms[c] for c in ("kpr", "adr", "survival", "kast", "multi")]
    parts.append(1 + pl.col("swing") / (4 * norms["swing_sd"]))
    hltv = sum(parts[1:], parts[0]) / len(parts)
    return table.with_columns(leetify.alias("leetify_like"), hltv.alias("hltv_like"))


def component_norms(tables: list[pl.DataFrame]) -> dict[str, float]:
    df = pl.concat(tables)
    norms = {c: float(df[c].mean() or 1.0) or 1.0 for c in ("kpr", "adr", "survival", "kast", "multi")}
    norms["swing_sd"] = float(df["swing"].std() or 1.0) or 1.0
    return norms
