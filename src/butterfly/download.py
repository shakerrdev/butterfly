from pathlib import Path

from huggingface_hub import snapshot_download

PRO_DATASET = "KEDevO/CounterQuant-CS2-Demos"


def download_pro_demos(map_name: str, out_dir: Path, years: list[str] | None = None, tiers: list[str] | None = None) -> Path:
    short = map_name.removeprefix("de_")
    years = years or ["*"]
    tiers = tiers or ["*"]
    patterns = [f"data/{year}/{tier}/*/*-{short}.dem" for year in years for tier in tiers]
    snapshot_download(
        repo_id=PRO_DATASET,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=out_dir,
    )
    return out_dir
