import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from demoparser2 import DemoParser

from butterfly.state import GI, GLOBAL_COLS, P, PI, PLAYER_COLS, SLOTS, STATE_LEN

TICKRATE = 64
SAMPLE_EVERY = 16
ROUND_TIME = 115.0
BOMB_TIME = 40.0
CT, T = 3, 2
SURRENDER_REASONS = {17, 18}

TICK_PROPS = [
    "X", "Y", "Z", "health", "armor_value", "has_helmet", "is_alive", "team_num", "balance",
    "current_equip_value", "inventory", "has_defuser", "is_defusing", "team_clan_name",
]

WEAPON_CLASS = {
    **dict.fromkeys(["AK-47", "M4A4", "M4A1-S", "FAMAS", "Galil AR", "AUG", "SG 553", "SSG 08"], "rifle"),
    **dict.fromkeys(["AWP", "G3SG1", "SCAR-20"], "awp"),
    **dict.fromkeys(["MAC-10", "MP9", "MP7", "MP5-SD", "UMP-45", "P90", "PP-Bizon"], "smg"),
    **dict.fromkeys(["Nova", "XM1014", "MAG-7", "Sawed-Off", "M249", "Negev"], "heavy"),
    **dict.fromkeys(["Desert Eagle", "R8 Revolver", "Five-SeveN", "Tec-9", "CZ75-Auto", "P250", "Dual Berettas"],
                    "pistol_up"),
}
GRENADES = {
    "Smoke Grenade": "smokes", "Flashbang": "flashes", "High Explosive Grenade": "he",
    "Molotov": "molotovs", "Incendiary Grenade": "molotovs",
}


@dataclass
class Round:
    number: int
    start: int
    live: int
    end: int
    winner: int
    reason: int


@dataclass
class ParsedMatch:
    meta: dict
    states: pl.DataFrame
    roster: pl.DataFrame
    events: pl.DataFrame

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "match.json").write_text(json.dumps(self.meta, indent=2, ensure_ascii=False))
        self.states.write_parquet(out / "states.parquet")
        self.roster.write_parquet(out / "roster.parquet")
        self.events.write_parquet(out / "events.parquet")


def _event(parser: DemoParser, name: str, player: list[str] | None = None) -> pl.DataFrame:
    try:
        df = parser.parse_event(name, player=player or [])
    except Exception:
        return pl.DataFrame()
    if df is None or len(df) == 0:
        return pl.DataFrame()
    return pl.from_pandas(df)


def _sid(value) -> str | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
        s = str(int(value))
    except (TypeError, ValueError):
        s = str(value)
    return None if s in ("0", "") else s


def find_rounds(parser: DemoParser) -> list[Round]:
    ends = _event(parser, "round_end")
    freezes = _event(parser, "round_freeze_end")
    starts = _event(parser, "round_start")
    if ends.is_empty() or freezes.is_empty():
        return []
    freeze_ticks = sorted(freezes["tick"].to_list())
    start_ticks = sorted(starts["tick"].to_list()) if not starts.is_empty() else []
    counts = parser.parse_ticks(["total_rounds_played", "is_warmup_period"], ticks=freeze_ticks)
    played = pl.from_pandas(counts).group_by("tick").agg(
        pl.col("total_rounds_played").max(), pl.col("is_warmup_period").max()).sort("tick")
    by_number: dict[int, Round] = {}
    end_rows = ends.sort("tick").to_dicts()
    for row in played.iter_rows(named=True):
        if row["is_warmup_period"]:
            continue
        live = row["tick"]
        end = next((e for e in end_rows if e["tick"] > live), None)
        if end is None:
            continue
        start = max([s for s in start_ticks if s <= live], default=live)
        number = int(row["total_rounds_played"]) + 1
        by_number[number] = Round(number, start, live, int(end["tick"]), int(end["winner"]), int(end["reason"]))
    rounds = [by_number[n] for n in sorted(by_number)]
    return [r for i, r in enumerate(rounds) if r.number == i + 1]


def _inventory(items) -> dict[str, float]:
    out = defaultdict(float)
    for item in items if items is not None else []:
        cls = WEAPON_CLASS.get(item)
        if cls:
            out[cls] = 1.0
        nade = GRENADES.get(item)
        if nade:
            out[nade] += 1.0
        if item == "C4 Explosive":
            out["has_bomb"] = 1.0
    return out


class _UtilityTracker:
    def __init__(self, parser: DemoParser):
        self.spans: dict[str, list[tuple[int, int, str, float, float]]] = {"smoke": [], "fire": []}
        for kind, start_ev, stop_ev in (("smoke", "smokegrenade_detonate", "smokegrenade_expired"),
                                        ("fire", "inferno_startburn", "inferno_expire")):
            starts, stops = _event(parser, start_ev), _event(parser, stop_ev)
            stop_at = {}
            if not stops.is_empty():
                for r in stops.iter_rows(named=True):
                    stop_at.setdefault(r["entityid"], r["tick"])
            if starts.is_empty():
                continue
            for r in starts.iter_rows(named=True):
                sid = _sid(r.get("user_steamid"))
                if sid is None:
                    continue
                end = stop_at.get(r["entityid"], r["tick"] + 22 * TICKRATE)
                self.spans[kind].append((r["tick"], end, sid, float(r["x"]), float(r["y"])))

    def active(self, kind: str, sid: str, tick: int):
        best = None
        for start, end, owner, x, y in self.spans[kind]:
            if owner == sid and start <= tick < end and (best is None or start > best[0]):
                best = (start, x, y)
        return best


class _Blinds:
    def __init__(self, blinds: pl.DataFrame):
        self.by_victim: dict[str, list[tuple[int, float]]] = defaultdict(list)
        if not blinds.is_empty():
            for r in blinds.iter_rows(named=True):
                sid = _sid(r["user_steamid"])
                if sid:
                    self.by_victim[sid].append((r["tick"], float(r["blind_duration"])))

    def remaining(self, sid: str, tick: int) -> float:
        left = 0.0
        for start, dur in self.by_victim.get(sid, []):
            if start <= tick:
                left = max(left, dur - (tick - start) / TICKRATE)
        return left


def _sample_ticks(r: Round) -> list[tuple[str, int]]:
    ticks = [("pre", r.start), ("live", r.live)]
    ticks += [("live", t) for t in range(r.live + SAMPLE_EVERY, r.end, SAMPLE_EVERY)]
    ticks.append(("end", r.end))
    return ticks


def parse_demo(path: Path, order_key: str | None = None, draw_possible: bool | None = None) -> ParsedMatch:
    parser = DemoParser(str(path))
    header = parser.parse_header()
    rounds = find_rounds(parser)
    if not rounds:
        raise ValueError(f"no rounds found in {path}")

    samples = [(r, kind, tick) for r in rounds for kind, tick in _sample_ticks(r)]
    all_ticks = sorted({t for _, _, t in samples})
    ticks = pl.from_pandas(parser.parse_ticks(TICK_PROPS, ticks=all_ticks))
    ticks = ticks.with_columns(pl.col("steamid").map_elements(_sid, return_dtype=pl.String))
    by_tick = {t: df for (t,), df in ticks.partition_by("tick", as_dict=True).items()}

    plants = _event(parser, "bomb_planted", player=["X", "Y", "last_place_name"])
    defuses = _event(parser, "bomb_defused")
    hurts, blinds = _event(parser, "player_hurt"), _event(parser, "player_blind")
    deaths = _event(parser, "player_death")
    util = _UtilityTracker(parser)
    blind_state = _Blinds(blinds)

    roster_rows, slot_of = [], {}
    team_of: dict[str, int] = {}
    team_names = {1: "", 2: ""}
    for r in rounds:
        frame = by_tick.get(r.live)
        if frame is None:
            continue
        slots = {}
        for side, base in ((CT, 0), (T, 5)):
            rows = frame.filter((pl.col("team_num") == side) & pl.col("steamid").is_not_null())
            rows = rows.sort(["is_alive", "steamid"], descending=[True, False]).head(5).sort("steamid")
            for i, sid in enumerate(rows["steamid"].to_list()):
                slots[sid] = base + i
        ct_team = _ct_team(team_of, slots)
        for sid, slot in slots.items():
            team = ct_team if slot < 5 else 3 - ct_team
            team_of[sid] = team
            row = frame.filter(pl.col("steamid") == sid).row(0, named=True)
            if row["team_clan_name"] and not team_names[team]:
                team_names[team] = row["team_clan_name"]
            roster_rows.append({"round": r.number, "slot": slot, "steamid": sid, "name": row["name"], "team": team})
        slot_of[r.number] = slots

    roster = pl.DataFrame(roster_rows)
    ct_team_of = {row["round"]: row["team"] for row in roster.filter(pl.col("slot") < 5).unique("round").iter_rows(named=True)}

    score = {1: 0, 2: 0}
    streak = {1: 0, 2: 0}
    plant_at = {}
    if not plants.is_empty():
        for p in plants.iter_rows(named=True):
            plant_at[p["tick"]] = p
    sites = _site_ids(plants)

    state_rows = []
    for r in rounds:
        slots = slot_of.get(r.number, {})
        ct_team = ct_team_of.get(r.number, 1)
        if r.number == 13 or (r.number > 24 and (r.number - 25) % 3 == 0):
            streak = {1: 0, 2: 0}
        round_plant = next((p for t, p in sorted(plant_at.items()) if r.live <= t <= r.end), None)
        for kind, tick in _sample_ticks(r):
            frame = by_tick.get(tick)
            if frame is None:
                continue
            x = np.zeros(STATE_LEN)
            g = x[: len(GLOBAL_COLS)]
            g[GI["ct_score"]] = score[ct_team]
            g[GI["t_score"]] = score[3 - ct_team]
            g[GI["round_num"]] = r.number
            g[GI["ct_loss_streak"]] = min(streak[ct_team], 4)
            g[GI["t_loss_streak"]] = min(streak[3 - ct_team], 4)
            elapsed = (tick - r.live) / TICKRATE if kind != "pre" else 0.0
            g[GI["live"]] = 0.0 if kind == "pre" else 1.0
            g[GI["time_left"]] = ROUND_TIME - elapsed if kind != "pre" else ROUND_TIME
            if round_plant and round_plant["tick"] <= tick and kind != "pre":
                g[GI["bomb_planted"]] = 1.0
                g[GI["bomb_site"]] = sites.get(round_plant["site"], 0)
                bomb_left = max(0.0, BOMB_TIME - (tick - round_plant["tick"]) / TICKRATE)
                g[GI["bomb_time_left"]] = bomb_left
                g[GI["time_left"]] = bomb_left
            if kind == "end":
                g[GI["round_over"]] = 1.0
                g[GI["round_winner"]] = 1.0 if r.winner == CT else -1.0
            for row in frame.iter_rows(named=True):
                slot = slots.get(row["steamid"])
                if slot is None:
                    continue
                v = np.zeros(P)
                alive = True if kind == "pre" else bool(row["is_alive"])
                inv = _inventory(row["inventory"])
                v[PI["x"]], v[PI["y"]], v[PI["z"]] = row["X"] or 0.0, row["Y"] or 0.0, row["Z"] or 0.0
                v[PI["hp"]] = (row["health"] or 0) if kind != "pre" else 100
                v[PI["armor"]] = row["armor_value"] or 0
                v[PI["helmet"]] = float(bool(row["has_helmet"]))
                v[PI["alive"]] = float(alive)
                v[PI["flash"]] = blind_state.remaining(row["steamid"], tick) if alive else 0.0
                v[PI["money"]] = row["balance"] or 0
                v[PI["equip"]] = row["current_equip_value"] or 0
                for col in ("rifle", "awp", "smg", "heavy", "pistol_up", "smokes", "flashes", "he", "molotovs",
                            "has_bomb"):
                    v[PI[col]] = inv.get(col, 0.0)
                v[PI["has_kit"]] = float(bool(row["has_defuser"]))
                v[PI["defusing"]] = float(bool(row["is_defusing"]))
                if kind != "pre":
                    for util_kind in ("smoke", "fire"):
                        act = util.active(util_kind, row["steamid"], tick)
                        if act:
                            v[PI[f"{util_kind}_on"]] = 1.0
                            v[PI[f"{util_kind}_x"]], v[PI[f"{util_kind}_y"]] = act[1], act[2]
                start = len(GLOBAL_COLS) + slot * P
                x[start:start + P] = v
            state_rows.append((r.number, kind, tick, x))
        if r.winner in (CT, T):
            winner_team = ct_team if r.winner == CT else 3 - ct_team
            score[winner_team] += 1
            streak[winner_team] = 0
            streak[3 - winner_team] += 1

    late = _event(parser, "round_end")
    late = late.filter(pl.col("tick") > rounds[-1].end) if not late.is_empty() else late
    final_reason, final_winner = rounds[-1].reason, rounds[-1].winner
    if not late.is_empty():
        final_reason, final_winner = int(late["reason"][-1]), int(late["winner"][-1])
    surrender = final_reason in SURRENDER_REASONS
    if surrender:
        last_ct = ct_team_of.get(rounds[-1].number, 1)
        winner = last_ct if final_winner == CT else 3 - last_ct
    elif score[1] == score[2]:
        winner = 0
    else:
        winner = 1 if score[1] > score[2] else 2
    complete = surrender or max(score.values()) >= 13 or (score[1] == score[2] == 12)
    if draw_possible is None:
        draw_possible = score[1] == score[2] == 12 and len(rounds) == 24

    states = pl.DataFrame({
        "round": [s[0] for s in state_rows],
        "kind": [s[1] for s in state_rows],
        "tick": [s[2] for s in state_rows],
    })
    mat = np.stack([s[3] for s in state_rows])
    names = list(GLOBAL_COLS) + [f"s{slot}_{c}" for slot in range(SLOTS) for c in PLAYER_COLS]
    states = states.hstack(pl.DataFrame(mat, schema=names))

    events = _events(rounds, hurts, blinds, plants, defuses, deaths, sites)
    meta = {
        "demo": path.name,
        "map": header.get("map_name"),
        "patch": header.get("patch_version"),
        "order_key": order_key or path.stem,
        "rounds": len(rounds),
        "score": {"1": score[1], "2": score[2]},
        "teams": {"1": team_names[1], "2": team_names[2]},
        "winner": winner,
        "complete": bool(complete),
        "surrender": surrender,
        "draw_possible": bool(draw_possible),
    }
    return ParsedMatch(meta, states, roster, events)


def _ct_team(team_of: dict[str, int], slots: dict[str, int]) -> int:
    votes = {1: 0, 2: 0}
    for sid, slot in slots.items():
        if sid in team_of:
            known = team_of[sid]
            votes[known if slot < 5 else 3 - known] += 1
    return 2 if votes[2] > votes[1] else 1


def _site_ids(plants: pl.DataFrame) -> dict[int, int]:
    if plants.is_empty():
        return {}
    ids = {}
    for p in plants.iter_rows(named=True):
        place = str(p.get("user_last_place_name") or "")
        if "BombsiteA" in place:
            ids[p["site"]] = 1
        elif "BombsiteB" in place:
            ids[p["site"]] = 2
    for site in sorted(set(plants["site"].to_list())):
        ids.setdefault(site, 1 if 1 not in ids.values() else 2)
    return ids


def _events(rounds: list[Round], hurts, blinds, plants, defuses, deaths, sites: dict[int, int]) -> pl.DataFrame:
    rows = []

    def add(df: pl.DataFrame, kind: str, amount):
        if df.is_empty():
            return
        for r in df.iter_rows(named=True):
            rows.append({
                "tick": int(r["tick"]),
                "kind": kind,
                "attacker": _sid(r.get("attacker_steamid")),
                "victim": _sid(r.get("user_steamid")),
                "amount": float(amount(r)),
                "x": r.get("user_X"),
                "y": r.get("user_Y"),
                "site": sites.get(r["site"]) if "site" in r and kind == "plant" else None,
                "assister": _sid(r.get("assister_steamid")) if kind == "death" else None,
                "flash_assist": bool(r.get("assistedflash")) if kind == "death" else None,
            })

    add(hurts, "damage", lambda r: r["dmg_health"] or 0)
    add(blinds, "blind", lambda r: r["blind_duration"] or 0)
    add(plants, "plant", lambda r: 1)
    add(defuses, "defuse", lambda r: 1)
    add(deaths, "death", lambda r: 1)
    if not rows:
        return pl.DataFrame(schema={"tick": pl.Int64, "kind": pl.String, "attacker": pl.String,
                                    "victim": pl.String, "amount": pl.Float64, "x": pl.Float64,
                                    "y": pl.Float64, "site": pl.Int64, "assister": pl.String,
                                    "flash_assist": pl.Boolean, "round": pl.Int64})
    df = pl.DataFrame(rows, schema_overrides={"x": pl.Float64, "y": pl.Float64, "site": pl.Int64,
                                              "assister": pl.String, "flash_assist": pl.Boolean}).sort("tick")
    bounds = [(r.number, r.start, r.end) for r in rounds]

    def round_of(t: int) -> int:
        for n, start, end in bounds:
            if start <= t <= end:
                return n
        return 0

    return df.with_columns(pl.col("tick").map_elements(round_of, return_dtype=pl.Int64).alias("round")).filter(
        pl.col("round") > 0)
