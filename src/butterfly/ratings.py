from collections import defaultdict

import numpy as np
import polars as pl

from butterfly.credit import Interval, damage_weight, round_credit
from butterfly.dataset import Match
from butterfly.model import WinModel

CATEGORIES = ["duels", "positioning", "utility", "economy", "objective", "shared"]


def _intervals(match: Match, ticks: list[int], slots: dict[str, int]) -> list[Interval]:
    ev = match.events.filter((pl.col("tick") > ticks[0]) & (pl.col("tick") <= ticks[-1]))
    out = []
    rows = ev.to_dicts()
    i = 0
    for t0, t1 in zip(ticks, ticks[1:]):
        interval = Interval()
        while i < len(rows) and rows[i]["tick"] <= t1:
            e = rows[i]
            i += 1
            if e["tick"] <= t0:
                continue
            victim = slots.get(e["victim"])
            if e["kind"] in ("plant", "defuse"):
                if victim is not None:
                    interval.objective = victim
                continue
            attacker = slots.get(e["attacker"])
            if victim is None or attacker is None or attacker == victim:
                continue
            w = damage_weight(dmg=e["amount"]) if e["kind"] == "damage" else damage_weight(blind_seconds=e["amount"])
            interval.damage.setdefault(victim, {})
            interval.damage[victim][attacker] = interval.damage[victim].get(attacker, 0.0) + w
        out.append(interval)
    return out


def _team_frame(v_ct: float, ct_team: int, team: int) -> float:
    return v_ct if team == ct_team else 1.0 - v_ct


def rate_match(model: WinModel, match: Match, n_perm: int = 16, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    draw = 1.0 if match.meta.get("draw_possible") else 0.0

    def predict(states: np.ndarray) -> np.ndarray:
        return model.predict(states, draw)

    info = match.info.with_row_index("i")
    ct_team_by_round = {r: t for r, t in match.roster.filter(pl.col("slot") < 5).unique("round")
                        .select("round", "team").iter_rows()}
    totals: dict[str, float] = defaultdict(float)
    cats: dict[tuple[str, str], float] = defaultdict(float)
    rounds_played: dict[str, int] = defaultdict(int)
    timeline = []
    prev_end: tuple[float, int] | None = None

    for (rnd,), part in info.group_by("round", maintain_order=True):
        idx = part["i"].to_numpy()
        ticks = part["tick"].to_list()
        kinds = part["kind"].to_list()
        roster = match.roster.filter(pl.col("round") == rnd)
        slots = dict(zip(roster["steamid"].to_list(), roster["slot"].to_list()))
        sid_of = {s: sid for sid, s in slots.items()}
        team_of = dict(zip(roster["steamid"].to_list(), roster["team"].to_list()))
        ct_team = ct_team_by_round.get(rnd, 1)
        states = match.states[idx]
        buy_steps = {n for n in range(len(kinds) - 1) if kinds[n] == "pre"}
        credit, cat, _ = round_credit(predict, states, buy_steps, _intervals(match, ticks, slots), n_perm, seed + rnd)

        ends = predict(states[[0, -1]])
        if prev_end is not None:
            v_prev, prev_ct = prev_end
            for team in (1, 2):
                delta = _team_frame(ends[0], ct_team, team) - _team_frame(v_prev, prev_ct, team)
                members = [sid for sid, t in team_of.items() if t == team] or []
                for sid in members:
                    totals[sid] += delta / len(members)
                    cats[(sid, "shared")] += delta / len(members)
        prev_end = (float(ends[1]), ct_team)

        for slot, value in enumerate(credit):
            sid = sid_of.get(slot)
            if sid is None:
                continue
            totals[sid] += float(value)
            rounds_played[sid] += 1
            timeline.append({"round": int(rnd), "steamid": sid, "credit": float(value)})
        for (slot, category), value in cat.items():
            sid = sid_of.get(slot)
            if sid is not None:
                cats[(sid, category)] += float(value)

    winner = match.meta["winner"]
    if prev_end is not None:
        v_last, last_ct = prev_end
        last_roster = match.roster.filter(pl.col("round") == match.roster["round"].max())
        for team in (1, 2):
            outcome = 0.5 if winner == 0 else float(winner == team)
            delta = outcome - _team_frame(v_last, last_ct, team)
            members = last_roster.filter(pl.col("team") == team)["steamid"].to_list()
            for sid in members:
                totals[sid] += delta / len(members)
                cats[(sid, "shared")] += delta / len(members)

    names = dict(match.roster.select("steamid", "name").unique("steamid").iter_rows())
    teams = dict(match.roster.sort("round").select("steamid", "team").unique("steamid", keep="last").iter_rows())
    rows = []
    for sid, total in totals.items():
        n = max(rounds_played.get(sid, 0), 1)
        row = {"steamid": sid, "name": names.get(sid, ""), "team": teams.get(sid, 0),
               "rounds": rounds_played.get(sid, 0), "wpa": total, "wpa_per_round": 100 * total / n,
               "rating": 1.0 + (total / n) / model.scale}
        for c in CATEGORIES:
            row[c] = 100 * cats.get((sid, c), 0.0) / n
        rows.append(row)
    return pl.DataFrame(rows).sort("rating", descending=True), pl.DataFrame(timeline)
