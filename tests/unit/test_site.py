"""The static results page is built from the committed artifacts only."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_site_builds_from_artifacts(tmp_path):
    index = _load_builder().build(tmp_path / "site", commit="0123456789abcdef")
    page = index.read_text(encoding="utf-8")
    # Headline numbers from the README's results table (mean ± sd over 3 seeds).
    assert "0.942 ± 0.006" in page
    assert "0.864 ± 0.009" in page
    assert "3 of 3" in page  # every non-private run's attack CI is above chance
    assert "0123456" in page
    # Every referenced figure and data file was copied next to the page.
    for ref in ["figures/privacy_utility.png", "figures/lira_roc.png", "data/lira.json",
                "data/dp_summary_statistics.json"]:
        assert ref in page
        assert (index.parent / ref).exists()
    # Nothing from data/ or checkpoints is published.
    published = {p.suffix for p in index.parent.rglob("*") if p.is_file()}
    assert published <= {".html", ".png", ".json", ".csv", ".md", ""}
