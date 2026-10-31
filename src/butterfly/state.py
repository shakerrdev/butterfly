from dataclasses import dataclass

import numpy as np

PLAYER_COLS = [
    "x", "y", "z",
    "hp", "armor", "helmet", "alive", "flash",
    "money", "equip", "rifle", "awp", "smg", "heavy", "pistol_up",
    "smokes", "flashes", "he", "molotovs", "has_bomb", "has_kit", "defusing",
    "smoke_on", "smoke_x", "smoke_y", "fire_on", "fire_x", "fire_y",
]
GLOBAL_COLS = [
    "ct_score", "t_score", "round_num", "live", "time_left",
    "bomb_planted", "bomb_site", "bomb_time_left",
    "round_over", "round_winner", "ct_loss_streak", "t_loss_streak",
]

P = len(PLAYER_COLS)
G = len(GLOBAL_COLS)
SLOTS = 10
CT_SLOTS = range(0, 5)
T_SLOTS = range(5, 10)
STATE_LEN = G + SLOTS * P

PLAYER_GROUPS = {
    "body": ["hp", "armor", "helmet", "alive", "flash"],
    "move": ["x", "y", "z"],
    "kit": ["money", "equip", "rifle", "awp", "smg", "heavy", "pistol_up", "smokes", "flashes", "he",
            "molotovs", "has_bomb", "has_kit", "defusing"],
    "util": ["smoke_on", "smoke_x", "smoke_y", "fire_on", "fire_x", "fire_y"],
}
BOMB_COLS = ["bomb_planted", "bomb_site"]
CONTEXT_COLS = [c for c in GLOBAL_COLS if c not in BOMB_COLS]

PI = {c: i for i, c in enumerate(PLAYER_COLS)}
GI = {c: i for i, c in enumerate(GLOBAL_COLS)}

assert sorted(sum(PLAYER_GROUPS.values(), [])) == sorted(PLAYER_COLS)


def player_index(slot: int, col: str) -> int:
    return G + slot * P + PI[col]


@dataclass(frozen=True)
class Group:
    kind: str
    slot: int | None
    index: np.ndarray


def all_groups() -> list[Group]:
    groups = [
        Group("context", None, np.array([GI[c] for c in CONTEXT_COLS])),
        Group("bomb", None, np.array([GI[c] for c in BOMB_COLS])),
    ]
    for slot in range(SLOTS):
        for kind, cols in PLAYER_GROUPS.items():
            groups.append(Group(kind, slot, np.array([player_index(slot, c) for c in cols])))
    return groups


GROUPS = all_groups()


def side_of(slot: int) -> int:
    return 3 if slot < 5 else 2


def split(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=np.float64)
    return states[..., :G], states[..., G:].reshape(*states.shape[:-1], SLOTS, P)
