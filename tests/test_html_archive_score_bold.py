"""HTML rendering should bold the winner's score in post_match / end_of_day."""
from __future__ import annotations

import pytest

from src import html_archive


@pytest.fixture
def isolated_index(tmp_path, monkeypatch):
    idx = tmp_path / "index.html"
    monkeypatch.setattr(html_archive, "INDEX", idx)
    monkeypatch.setattr(html_archive, "DOCS_DIR", tmp_path)
    return idx


def test_post_match_bolds_winner_score(isolated_index):
    body = (
        "MI beat PBKS by 6 wickets\n"
        "PBKS 200/8 (20.0)  ·  MI 203/4 (18.2)\n"
        "\n"
        "Pre-match prediction: incorrect"
    )
    html_archive.upsert_message("2026-05-14", "post_match_1", "2026-05-14T20:00:00-07:00", body)
    html = isolated_index.read_text()
    # Winner segment is bolded; loser segment is not
    assert "<strong>MI 203/4 (18.2)</strong>" in html
    assert "PBKS 200/8 (20.0)  ·  " in html
    assert "<strong>PBKS 200/8 (20.0)</strong>" not in html


def test_end_of_day_bolds_winner_score_for_each_match(isolated_index):
    body = (
        "IPL 2026 - Thursday, May 14 - Day recap\n"
        "\n"
        "MI beat PBKS by 6 wickets\n"
        "PBKS 200/8 (20.0)  ·  MI 203/4 (18.2)\n"
        "GT beat SRH by 82 runs\n"
        "GT 220/3 (20.0)  ·  SRH 138/9 (20.0)\n"
        "\n"
        "Season to date prediction: 2 of 4 correct (50%)"
    )
    html_archive.upsert_message("2026-05-14", "end_of_day", "2026-05-14T21:00:00-07:00", body)
    html = isolated_index.read_text()
    assert "<strong>MI 203/4 (18.2)</strong>" in html
    assert "<strong>GT 220/3 (20.0)</strong>" in html
    # Losers stay unbolded
    assert "<strong>PBKS 200/8 (20.0)</strong>" not in html
    assert "<strong>SRH 138/9 (20.0)</strong>" not in html
