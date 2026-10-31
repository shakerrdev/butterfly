from types import SimpleNamespace

from butterfly.download import list_pro_demos, summarize

TREE = [
    SimpleNamespace(path="data/2026/tier1/2401337"),
    SimpleNamespace(path="data/2026/tier1/2401337/vitality-vs-g2-ancient.dem", size=300_000_000),
    SimpleNamespace(path="data/2026/tier1/2401337/vitality-vs-g2-inferno.dem", size=280_000_000),
    SimpleNamespace(path="data/2026/tier1/2401400/spirit-vs-mouz-ancient.dem", size=250_000_000),
    SimpleNamespace(path="data/2026/tier2/2401500/a-vs-b-ancient.dem", size=200_000_000),
    SimpleNamespace(path="data/2025/tier1/2300000/c-vs-d-ancient.dem", size=100_000_000),
]


class FakeApi:
    def __init__(self):
        self.roots = []

    def list_repo_tree(self, repo_id, path_in_repo, recursive, repo_type):
        self.roots.append(path_in_repo)
        return [e for e in TREE if e.path.startswith(path_in_repo + "/")]


def test_lists_only_matching_map_year_and_tier():
    api = FakeApi()
    files = list_pro_demos("de_ancient", ["2026"], ["tier1"], api=api)
    assert api.roots == ["data/2026/tier1"]
    assert [f.path for f in files] == [
        "data/2026/tier1/2401337/vitality-vs-g2-ancient.dem",
        "data/2026/tier1/2401400/spirit-vs-mouz-ancient.dem",
    ]


def test_summary_groups_by_year_and_tier():
    files = list_pro_demos("de_ancient", api=FakeApi())
    assert summarize(files).splitlines() == [
        "2025/tier1: 1 demos, 0.1 GB",
        "2026/tier1: 2 demos, 0.6 GB",
        "2026/tier2: 1 demos, 0.2 GB",
        "total: 4 demos, 0.8 GB (avg 212 MB)",
    ]
