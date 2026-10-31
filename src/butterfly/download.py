from collections import defaultdict
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

PRO_DATASET = "KEDevO/CounterQuant-CS2-Demos"


@dataclass
class DemoFile:
    path: str
    size: int

    @property
    def year_tier(self) -> str:
        parts = self.path.split("/")
        return f"{parts[1]}/{parts[2]}" if len(parts) > 3 else "?"


def patterns(map_name: str, years: list[str] | None, tiers: list[str] | None) -> list[str]:
    short = map_name.removeprefix("de_")
    return [f"data/{year}/{tier}/*/*-{short}.dem" for year in years or ["*"] for tier in tiers or ["*"]]


def _roots(years: list[str] | None, tiers: list[str] | None) -> list[str]:
    if not years:
        return ["data"]
    if not tiers:
        return [f"data/{y}" for y in years]
    return [f"data/{y}/{t}" for y in years for t in tiers]


def list_pro_demos(map_name: str, years: list[str] | None = None, tiers: list[str] | None = None,
                   api: HfApi | None = None) -> list[DemoFile]:
    api = api or HfApi()
    wanted = patterns(map_name, years, tiers)
    found = []
    for root in _roots(years, tiers):
        for entry in api.list_repo_tree(PRO_DATASET, path_in_repo=root, recursive=True, repo_type="dataset"):
            size = getattr(entry, "size", None)
            if size is not None and any(fnmatch(entry.path, p) for p in wanted):
                found.append(DemoFile(entry.path, int(size)))
    return sorted(found, key=lambda f: f.path)


def summarize(files: list[DemoFile]) -> str:
    groups = defaultdict(list)
    for f in files:
        groups[f.year_tier].append(f.size)
    lines = [f"{key}: {len(sizes)} demos, {_gb(sum(sizes))}" for key, sizes in sorted(groups.items())]
    total = sum(f.size for f in files)
    avg = total / len(files) if files else 0
    lines.append(f"total: {len(files)} demos, {_gb(total)} (avg {avg / 1e6:.0f} MB)")
    return "\n".join(lines)


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def download_pro_demos(map_name: str, out_dir: Path, years: list[str] | None = None,
                       tiers: list[str] | None = None) -> Path:
    snapshot_download(
        repo_id=PRO_DATASET,
        repo_type="dataset",
        allow_patterns=patterns(map_name, years, tiers),
        local_dir=out_dir,
    )
    return out_dir
