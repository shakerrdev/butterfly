import numpy as np
import pytest

from butterfly.shapley import EXACT_MAX_GROUPS, step_shapley
from butterfly.state import GI, STATE_LEN, player_index


def linear(states: np.ndarray) -> np.ndarray:
    return 0.5 + 0.01 * states[:, player_index(0, "hp")] - 0.01 * states[:, player_index(5, "hp")]


def interacting(states: np.ndarray) -> np.ndarray:
    a = states[:, player_index(0, "alive")]
    b = states[:, player_index(1, "alive")]
    return a * b


def test_linear_model_gives_each_group_its_own_effect():
    x0 = np.zeros(STATE_LEN)
    x1 = x0.copy()
    x1[player_index(0, "hp")] = 10
    x1[player_index(5, "hp")] = 4
    [(changed, phi, v0, v1)] = step_shapley(linear, [(x0, x1)])
    got = {(g.kind, g.slot): v for g, v in zip(changed, phi)}
    assert got == pytest.approx({("body", 0): 0.10, ("body", 5): -0.04})
    assert v1 - v0 == pytest.approx(phi.sum())


def test_interaction_is_split_evenly():
    x0 = np.zeros(STATE_LEN)
    x1 = x0.copy()
    x1[player_index(0, "alive")] = 1
    x1[player_index(1, "alive")] = 1
    [(changed, phi, _, _)] = step_shapley(interacting, [(x0, x1)])
    assert phi == pytest.approx([0.5, 0.5])


def test_sampled_shapley_is_efficient_and_close():
    x0 = np.zeros(STATE_LEN)
    x1 = x0.copy()
    for slot in range(EXACT_MAX_GROUPS + 3):
        x1[player_index(slot, "hp")] = slot + 1
    x1[GI["time_left"]] = -5

    def model(s):
        return 0.002 * s[:, [player_index(i, "hp") for i in range(10)]].sum(1) + 0.01 * s[:, GI["time_left"]]

    [(changed, phi, v0, v1)] = step_shapley(model, [(x0, x1)], n_perm=8)
    assert phi.sum() == pytest.approx(v1 - v0)
    got = {(g.kind, g.slot): v for g, v in zip(changed, phi)}
    assert got[("context", None)] == pytest.approx(-0.05)
    assert got[("body", 3)] == pytest.approx(0.008)


def test_unchanged_step_has_no_groups():
    x = np.zeros(STATE_LEN)
    [(changed, phi, v0, v1)] = step_shapley(linear, [(x, x.copy())])
    assert changed == [] and len(phi) == 0 and v0 == v1
