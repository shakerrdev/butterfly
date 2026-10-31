from dataclasses import dataclass
from itertools import combinations
from math import factorial
from typing import Callable

import numpy as np

from butterfly.state import GROUPS, Group

Predict = Callable[[np.ndarray], np.ndarray]

EXACT_MAX_GROUPS = 6


@dataclass
class StepPlan:
    changed: list[Group]
    start: int
    stop: int
    exact: bool
    subsets: list[tuple[int, ...]] | None = None
    perms: np.ndarray | None = None


def changed_groups(x0: np.ndarray, x1: np.ndarray) -> list[Group]:
    diff = x0 != x1
    return [g for g in GROUPS if diff[g.index].any()]


def _mix(x0: np.ndarray, x1: np.ndarray, groups: list[Group], members) -> np.ndarray:
    x = x0.copy()
    for i in members:
        idx = groups[i].index
        x[idx] = x1[idx]
    return x


def plan_rows(x0: np.ndarray, x1: np.ndarray, rng: np.random.Generator, n_perm: int, offset: int) -> tuple[StepPlan, list[np.ndarray]]:
    changed = changed_groups(x0, x1)
    k = len(changed)
    if k == 0:
        return StepPlan(changed, offset, offset, True, []), []
    if k <= EXACT_MAX_GROUPS:
        subsets = [s for r in range(k + 1) for s in combinations(range(k), r)]
        rows = [_mix(x0, x1, changed, s) for s in subsets]
        return StepPlan(changed, offset, offset + len(rows), True, subsets=subsets), rows
    half = max(1, n_perm // 2)
    forward = np.array([rng.permutation(k) for _ in range(half)])
    perms = np.concatenate([forward, forward[:, ::-1]])
    rows = []
    for perm in perms:
        x = x0.copy()
        rows.append(x.copy())
        for i in perm:
            idx = changed[i].index
            x[idx] = x1[idx]
            rows.append(x.copy())
    return StepPlan(changed, offset, offset + len(rows), False, perms=perms), rows


def solve(plan: StepPlan, values: np.ndarray) -> np.ndarray:
    k = len(plan.changed)
    phi = np.zeros(k)
    if k == 0:
        return phi
    v = values[plan.start:plan.stop]
    if plan.exact:
        lookup = {s: v[i] for i, s in enumerate(plan.subsets)}
        weights = [factorial(r) * factorial(k - r - 1) / factorial(k) for r in range(k)]
        for s, val in lookup.items():
            members = set(s)
            for i in range(k):
                if i in members:
                    continue
                with_i = tuple(sorted(members | {i}))
                phi[i] += weights[len(s)] * (lookup[with_i] - val)
        return phi
    per = k + 1
    for p, perm in enumerate(plan.perms):
        chain = v[p * per:(p + 1) * per]
        phi[perm] += np.diff(chain)
    return phi / len(plan.perms)


def step_shapley(predict: Predict, pairs: list[tuple[np.ndarray, np.ndarray]], n_perm: int = 16,
                 seed: int = 0, chunk: int = 200_000) -> list[tuple[list[Group], np.ndarray, float, float]]:
    rng = np.random.default_rng(seed)
    plans: list[StepPlan] = []
    rows: list[np.ndarray] = []
    for x0, x1 in pairs:
        plan, r = plan_rows(x0, x1, rng, n_perm, len(rows))
        plans.append(plan)
        rows.extend(r)
    values = np.empty(len(rows))
    for i in range(0, len(rows), chunk):
        values[i:i + chunk] = predict(np.stack(rows[i:i + chunk]))
    ends = predict(np.stack([x for pair in pairs for x in pair])) if pairs else np.empty(0)
    out = []
    for n, plan in enumerate(plans):
        out.append((plan.changed, solve(plan, values), float(ends[2 * n]), float(ends[2 * n + 1])))
    return out
