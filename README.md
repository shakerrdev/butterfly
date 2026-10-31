# butterfly

A CS2 player rating built on one question: how much did each player change their team's chance of winning the **match**?

HLTV 3.0 and Leetify only move win probability on kills, and only within the round. butterfly learns P(match win) from the whole game state every quarter second. That state covers positions, HP, weapons, utility in hand and on the map, money, the score and the bomb. Every change in that probability is then split between the players whose state change caused it. So a lurk that pulls rotations, a smoke that wins a retake, or killing a saved AWP all count. That last one counts because the model sees the AWP's value carrying into the next round.

## How it works

1. **Parse** (`parse`): each demo becomes a match folder.
   - The state is sampled at each round's freeze start, at freeze end, every 16 ticks while the round is live, and at the round's end.
   - Each player has four state groups: body (HP, armour, alive, flashed), position, kit (money, weapons, grenades, bomb, kit) and utility on the map (their active smoke or fire).
   - Global context covers the score, time, bomb and loss streaks.
2. **Model** (`train`): LightGBM on handcrafted team features predicts whether the team currently on CT wins the match. Draws count as half a win.
3. **Credit** (`rate`): each step's change in win probability is split over the state groups that changed, using Shapley values.
   - Exact when 6 or fewer groups changed; otherwise sampled permutations.
   - Your own changes are yours. Damage and flashes you took are paid to the opponents who caused them, weighted by damage and blind time.
   - Plants and defuses go to the planter or defuser.
   - Time and round transitions are shared by the team.
   - The result is zero-sum: one team's players add up to exactly what the other team's players lose.
4. **Rating**: win probability added per round, a headline rating centred on 1.00, and a breakdown into duels, positioning, utility, economy, objective and shared.
5. **Benchmark** (`benchmark`): checks whether the rating predicts later matches better than the alternatives.
   - The alternatives are reimplementations on the same demos: a Leetify-style kill swing (killer 35 / damage 30 / flash 15 / trade 20) and an HLTV-3.0-like composite (kills, damage, survival, KAST, multi-kills, round swing).
   - Each player's ratings from earlier matches predict who wins the held-out later matches.
   - It also reports split-half stability for each rating.

## Setup

```sh
uv sync
```

## Pro data (Ancient first)

Pro demos come from the Hugging Face dataset `KEDevO/CounterQuant-CS2-Demos` (HLTV Tier 1–2, 2024 onward, CC BY 4.0). The whole dataset is several TB, so download one year or tier at a time. Parse it, delete the demos, then fetch the next batch:

```sh
uv run butterfly download --map de_ancient --years 2026 --tiers tier1 --out D:/butterfly/demos/pro
uv run butterfly parse D:/butterfly/demos/pro --out data/parsed/pro --jobs 6 --delete
uv run butterfly train --data data/parsed/pro --map de_ancient --out models/ancient
uv run butterfly benchmark --data data/parsed/pro --map de_ancient --model models/ancient
```

The training split is by time: the latest 15% of matches are held out. Matches are ordered by HLTV match id, which rises over time. `train` writes `report.json`, which compares the model's log loss with a constant guess. It also sets the rating scale so that held-out players spread with a standard deviation of about 0.25 around 1.00.

## Your own games

```sh
uv run butterfly parse "D:/SteamLibrary/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays" --out data/parsed/pug --draws
uv run butterfly train --data data/parsed/pug --map de_ancient --base models/ancient --out models/ancient-pug --rounds 200 --learning-rate 0.02
uv run butterfly rate data/parsed/pug/<match> --model models/ancient-pug
```

- `--draws` marks matches that can end 12–12, which is MM competitive. Leave it off for Premier and FACEIT.
- Fine-tuning continues boosting from the pro model on your games. `rate` writes `ratings.parquet` and a per-round `credit.parquet` into the match folder.

## Known gaps

- At a side swap (round 12→13, OT halves), the change between rounds goes to the team's shared credit, because slots and economy reset there.
- Crouching, view direction, sound and line of sight aren't in the state yet.
- The HLTV-like baseline isn't eco-adjusted, and its weights are equal because HLTV's real ones aren't public.
