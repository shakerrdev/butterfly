import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl


def _demo_files(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        out.extend(sorted(p.rglob("*.dem")) if p.is_dir() else [p])
    return out


def _order_key(demo: Path) -> str:
    parent = demo.parent.name
    return f"{int(parent):012d}-{demo.stem}" if parent.isdigit() else f"{int(demo.stat().st_mtime):012d}-{demo.stem}"


def _parse_one(demo: Path, out: Path, draws: bool | None) -> str:
    from butterfly.parse import parse_demo

    match = parse_demo(demo, order_key=_order_key(demo), draw_possible=draws)
    match.save(out)
    return f"{demo.name}: {match.meta['map']} {match.meta['score']} {match.meta['rounds']} rounds"


def cmd_download(args) -> None:
    from butterfly.download import download_pro_demos

    download_pro_demos(args.map, args.out, args.years, args.tiers)
    print(f"demos in {args.out}")


def cmd_parse(args) -> None:
    demos = _demo_files(args.demos)
    todo = []
    for demo in demos:
        name = f"{demo.parent.name}-{demo.stem}" if demo.parent.name.isdigit() else demo.stem
        out = args.out / name
        if (out / "match.json").exists() and not args.force:
            continue
        todo.append((demo, out))
    print(f"{len(todo)} of {len(demos)} demos to parse")
    draws = True if args.draws else None
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_parse_one, demo, out, draws): demo for demo, out in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            demo = futures[fut]
            try:
                print(f"[{n}/{len(todo)}] {fut.result()}", flush=True)
            except Exception as e:
                print(f"[{n}/{len(todo)}] {demo.name}: failed: {e}", file=sys.stderr, flush=True)


def cmd_train(args) -> None:
    from butterfly.dataset import load_matches
    from butterfly.model import WinModel
    from butterfly.training import calibrate_scale, fit

    matches = load_matches(args.data, args.map)
    print(f"{len(matches)} complete {args.map or ''} matches")
    base = WinModel.load(args.base) if args.base else None
    params = {"learning_rate": args.learning_rate} if args.learning_rate else None
    model, report, valid = fit(matches, stride=args.stride, base=base, rounds=args.rounds, params=params,
                               target=args.target)
    held_out = [m for m, v in zip(matches, valid) if v][: args.calibrate]
    model.scale = calibrate_scale(model, held_out) if held_out else model.scale
    report["scale"] = model.scale
    model.save(args.out)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def cmd_rate(args) -> None:
    from butterfly.dataset import load_match
    from butterfly.model import WinModel
    from butterfly.ratings import rate_match

    model = WinModel.load(args.model)
    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(200)
    for path in args.matches:
        match = load_match(path)
        table, timeline = rate_match(model, match, n_perm=args.perms)
        table.write_parquet(path / "ratings.parquet")
        timeline.write_parquet(path / "credit.parquet")
        print(f"\n{match.meta['demo']}  {match.meta['teams']}  score {match.meta['score']}")
        print(table.select("name", "team", "rounds", "rating", "wpa_per_round", "duels", "positioning", "utility",
                           "economy", "objective", "shared").with_columns(pl.col(pl.Float64).round(2)))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="butterfly")
    sub = ap.add_subparsers(required=True)

    p = sub.add_parser("download", help="download pro demos for one map from Hugging Face")
    p.add_argument("--map", default="de_ancient")
    p.add_argument("--out", type=Path, default=Path("data/demos/pro"))
    p.add_argument("--years", nargs="*")
    p.add_argument("--tiers", nargs="*")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("parse", help="parse demos into match folders")
    p.add_argument("demos", type=Path, nargs="+")
    p.add_argument("--out", type=Path, default=Path("data/parsed/pro"))
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--draws", action="store_true", help="matches can end 12-12 (MM competitive)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_parse)

    p = sub.add_parser("train", help="train the match-win model")
    p.add_argument("--data", type=Path, default=Path("data/parsed/pro"))
    p.add_argument("--map", default="de_ancient")
    p.add_argument("--out", type=Path, default=Path("models/ancient"))
    p.add_argument("--base", type=Path, help="fine-tune from this model (pug data)")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--target", choices=["match", "round"], default="match")
    p.add_argument("--rounds", type=int, default=2000)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--calibrate", type=int, default=20, help="held-out matches used to scale the rating")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("rate", help="rate players in parsed matches")
    p.add_argument("matches", type=Path, nargs="+")
    p.add_argument("--model", type=Path, default=Path("models/ancient"))
    p.add_argument("--perms", type=int, default=16)
    p.set_defaults(func=cmd_rate)

    args = ap.parse_args(argv)
    args.func(args)
