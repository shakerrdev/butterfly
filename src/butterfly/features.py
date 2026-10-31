import warnings

import numpy as np

from butterfly.state import GI, PI, split

NEAR_SITE = 700.0
UTIL_NEAR_SITE = 900.0

Sites = dict[str, tuple[float, float]]


def _side_features(p: np.ndarray, sites: Sites | None) -> dict[str, np.ndarray]:
    alive = p[..., PI["alive"]] > 0.5
    a = alive.astype(np.float64)

    def total(col: str, alive_only: bool = True) -> np.ndarray:
        v = p[..., PI[col]]
        return (v * a).sum(-1) if alive_only else v.sum(-1)

    equip_alive = total("equip")
    money = total("money", alive_only=False)
    f = {
        "alive": a.sum(-1),
        "hp": total("hp"),
        "armor": ((p[..., PI["armor"]] > 0) & alive).sum(-1),
        "helmet": ((p[..., PI["helmet"]] > 0.5) & alive).sum(-1),
        "flashed": ((p[..., PI["flash"]] > 0.5) & alive).sum(-1),
        "money": money,
        "money_min": p[..., PI["money"]].min(-1),
        "equip": equip_alive,
        "wealth": money + equip_alive,
        "rifle": total("rifle"),
        "awp": total("awp"),
        "smg": total("smg"),
        "heavy": total("heavy"),
        "pistol_up": total("pistol_up"),
        "smokes": total("smokes"),
        "flashes": total("flashes"),
        "he": total("he"),
        "molotovs": total("molotovs"),
        "bomb": ((p[..., PI["has_bomb"]] > 0.5) & alive).any(-1).astype(np.float64),
        "kits": ((p[..., PI["has_kit"]] > 0.5) & alive).sum(-1),
        "defusing": ((p[..., PI["defusing"]] > 0.5) & alive).any(-1).astype(np.float64),
        "smoke_on": (p[..., PI["smoke_on"]] > 0.5).sum(-1),
        "fire_on": (p[..., PI["fire_on"]] > 0.5).sum(-1),
    }
    if sites:
        xy = p[..., [PI["x"], PI["y"]]]
        for name, (sx, sy) in sites.items():
            d = np.hypot(xy[..., 0] - sx, xy[..., 1] - sy)
            d_alive = np.where(alive, d, np.nan)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                f[f"dmin_{name}"] = np.where(alive.any(-1), np.nanmin(np.where(alive, d, np.inf), -1), np.nan)
                f[f"dmean_{name}"] = np.nanmean(d_alive, -1)
            f[f"near_{name}"] = ((d < NEAR_SITE) & alive).sum(-1)
            for util in ("smoke", "fire"):
                on = p[..., PI[f"{util}_on"]] > 0.5
                ud = np.hypot(p[..., PI[f"{util}_x"]] - sx, p[..., PI[f"{util}_y"]] - sy)
                f[f"{util}_near_{name}"] = (on & (ud < UTIL_NEAR_SITE)).sum(-1)
    return f


def _rounds_to_win(ct: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    played = ct + t
    ot_blocks = np.maximum(0, np.floor((played - 24) / 6) + 1) * (played >= 24)
    target = np.where(played >= 24, 16 + 3 * (ot_blocks - 1), 13)
    target = np.maximum(target, np.maximum(ct, t) + 1)
    return target - ct, target - t


def featurize(states: np.ndarray, sites: Sites | None = None, draw_possible: float = 0.0) -> tuple[np.ndarray, list[str]]:
    g, p = split(states)
    ct, t = p[..., :5, :], p[..., 5:, :]
    cols: dict[str, np.ndarray] = {}
    for name in ("ct_score", "t_score", "round_num", "live", "time_left", "bomb_planted", "bomb_site",
                 "bomb_time_left", "round_over", "round_winner", "ct_loss_streak", "t_loss_streak"):
        cols[name] = g[..., GI[name]]
    cols["score_diff"] = cols["ct_score"] - cols["t_score"]
    cols["ct_to_win"], cols["t_to_win"] = _rounds_to_win(cols["ct_score"], cols["t_score"])
    cols["draw_possible"] = np.full(g.shape[:-1], draw_possible)

    fct, ft = _side_features(ct, sites), _side_features(t, sites)
    for k, v in fct.items():
        cols[f"ct_{k}"] = v
    for k, v in ft.items():
        cols[f"t_{k}"] = v
    for k in ("alive", "hp", "wealth", "equip", "smokes", "flashes", "molotovs"):
        cols[f"diff_{k}"] = fct[k] - ft[k]

    if sites:
        names = list(sites)
        site_idx = np.clip(cols["bomb_site"].astype(int) - 1, 0, len(names) - 1)
        planted = cols["bomb_planted"] > 0.5
        for side in ("ct", "t"):
            stacked = np.stack([cols[f"{side}_dmin_{n}"] for n in names], -1)
            near = np.take_along_axis(stacked, site_idx[..., None], -1)[..., 0]
            cols[f"{side}_dmin_bomb"] = np.where(planted, near, np.nan)

    names = list(cols)
    return np.stack([cols[n] for n in names], -1).astype(np.float32), names
