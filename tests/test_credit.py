import numpy as np
import pytest

from butterfly.credit import Interval, round_credit
from butterfly.state import GI, STATE_LEN, player_index


def base_state() -> np.ndarray:
    x = np.zeros(STATE_LEN)
    for slot in range(10):
        x[player_index(slot, "alive")] = 1
        x[player_index(slot, "hp")] = 100
    return x


def ct_win(states: np.ndarray) -> np.ndarray:
    ct = states[:, [player_index(s, "alive") for s in range(5)]].sum(1)
    t = states[:, [player_index(s, "alive") for s in range(5, 10)]].sum(1)
    return 0.5 + 0.05 * (ct - t)


def test_kill_credit_goes_to_attacker_and_victim():
    x0 = base_state()
    x1 = x0.copy()
    x1[player_index(7, "alive")] = 0
    x1[player_index(7, "hp")] = 0
    interval = Interval(damage={7: {2: 100.0}})
    credit, cat, deltas = round_credit(ct_win, np.stack([x0, x1]), set(), [interval])
    assert deltas == pytest.approx([0.05])
    assert credit[2] == pytest.approx(0.05)
    assert credit[7] == pytest.approx(-0.05)
    assert credit[:5].sum() == pytest.approx(0.05)
    assert credit[5:].sum() == pytest.approx(-0.05)
    assert cat[(2, "duels")] == pytest.approx(0.05)


def test_shared_damage_splits_by_weight():
    x0 = base_state()
    x1 = x0.copy()
    x1[player_index(6, "alive")] = 0
    interval = Interval(damage={6: {0: 75.0, 1: 25.0}})
    credit, _, _ = round_credit(ct_win, np.stack([x0, x1]), set(), [interval])
    assert credit[0] == pytest.approx(0.0375)
    assert credit[1] == pytest.approx(0.0125)


def test_context_change_is_shared_by_alive_players():
    x0 = base_state()
    x0[player_index(4, "alive")] = 0
    x1 = x0.copy()
    x1[GI["time_left"]] = 10

    def model(s):
        return 0.5 + 0.004 * s[:, GI["time_left"]]

    credit, cat, _ = round_credit(model, np.stack([x0, x1]), set(), [Interval()])
    assert credit[:4] == pytest.approx([0.01] * 4)
    assert credit[4] == 0
    assert credit[5:] == pytest.approx([-0.008] * 5)


def test_plant_goes_to_planter():
    x0 = base_state()
    x1 = x0.copy()
    x1[GI["bomb_planted"]] = 1

    def model(s):
        return 0.6 - 0.3 * s[:, GI["bomb_planted"]]

    credit, cat, _ = round_credit(model, np.stack([x0, x1]), set(), [Interval(objective=8)])
    assert credit[8] == pytest.approx(0.3)
    assert credit[:5] == pytest.approx([-0.06] * 5)
    assert cat[(8, "objective")] == pytest.approx(0.3)
