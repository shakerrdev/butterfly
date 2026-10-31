from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from butterfly.shapley import Predict, step_shapley
from butterfly.state import SLOTS, player_index

CATEGORY = {"body": "duels", "move": "positioning", "util": "utility", "kit": "economy",
            "bomb": "objective", "context": "shared"}


@dataclass
class Interval:
    damage: dict[int, dict[int, float]] = field(default_factory=dict)
    objective: int | None = None


def _alive(x0: np.ndarray, x1: np.ndarray, slot: int) -> bool:
    i = player_index(slot, "alive")
    return x0[i] > 0.5 or x1[i] > 0.5


def _side_slots(ct: bool) -> range:
    return range(0, 5) if ct else range(5, 10)


def _split_even(credit: np.ndarray, cat: dict, slots, amount: float, x0, x1, category: str) -> None:
    alive = [s for s in slots if _alive(x0, x1, s)] or list(slots)
    for s in alive:
        credit[s] += amount / len(alive)
        cat[(s, category)] += amount / len(alive)


def assign(changed, phi: np.ndarray, x0: np.ndarray, x1: np.ndarray, interval: Interval,
           credit: np.ndarray, cat: dict, buy_step: bool = False) -> None:
    for g, value in zip(changed, phi):
        category = "economy" if buy_step and g.kind in ("kit", "body") else CATEGORY[g.kind]
        if g.kind == "context" or (g.kind == "bomb" and interval.objective is None):
            _split_even(credit, cat, _side_slots(True), value, x0, x1, category)
            _split_even(credit, cat, _side_slots(False), -value, x0, x1, category)
            continue
        owner = interval.objective if g.kind == "bomb" else g.slot
        own_ct = owner < 5
        own_value = value if own_ct else -value
        credit[owner] += own_value
        cat[(owner, category)] += own_value
        opponents = _side_slots(not own_ct)
        hurt = interval.damage.get(owner, {}) if g.kind == "body" else {}
        hurt = {s: w for s, w in hurt.items() if s in opponents and w > 0}
        if hurt:
            total = sum(hurt.values())
            for s, w in hurt.items():
                credit[s] -= own_value * w / total
                cat[(s, category)] -= own_value * w / total
        else:
            _split_even(credit, cat, opponents, -own_value, x0, x1, category)


def round_credit(predict: Predict, states: np.ndarray, buy_steps: set[int], intervals: list[Interval],
                 n_perm: int = 16, seed: int = 0) -> tuple[np.ndarray, dict, list[float]]:
    pairs = [(states[i], states[i + 1]) for i in range(len(states) - 1)]
    results = step_shapley(predict, pairs, n_perm=n_perm, seed=seed)
    credit = np.zeros(SLOTS)
    cat: dict = defaultdict(float)
    deltas = []
    for n, (changed, phi, v0, v1) in enumerate(results):
        x0, x1 = pairs[n]
        assign(changed, phi, x0, x1, intervals[n], credit, cat, buy_step=n in buy_steps)
        deltas.append(v1 - v0)
    return credit, dict(cat), deltas


def damage_weight(dmg: float = 0.0, blind_seconds: float = 0.0) -> float:
    return dmg + 25.0 * blind_seconds

