"""Tests that html_archive.upsert_message renders articles within a date in
descending (newest-first) order.

Today the rendering places newest day at top of the page, but within each
day the articles appear oldest-first — a mixed direction that's jarring to
read. Flip within-day to newest-first for consistency.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src import html_archive


@pytest.fixture
def isolated_index(tmp_path, monkeypatch):
    idx = tmp_path / "index.html"
    monkeypatch.setattr(html_archive, "INDEX", idx)
    monkeypatch.setattr(html_archive, "DOCS_DIR", tmp_path)
    return idx


def _generated_at_iso_list(idx_path: Path, date_iso: str) -> list[str]:
    """Return the data-generated values of articles in document order for
    the given date_iso section."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(idx_path.read_text(), "html.parser")
    details = soup.find("details", attrs={"data-day": date_iso})
    assert details is not None, f"no <details> for {date_iso}"
    return [a.get("data-generated", "") for a in details.find_all("article", recursive=False)]


def test_articles_within_a_date_render_newest_first(isolated_index):
    html_archive.upsert_message("2026-05-13", "morning",
                                "2026-05-13T00:01:13-07:00", "morning body")
    html_archive.upsert_message("2026-05-13", "toss_1",
                                "2026-05-13T08:02:51-07:00", "toss body")
    html_archive.upsert_message("2026-05-13", "powerplay_1_1",
                                "2026-05-13T08:46:20-07:00", "PP1 body")

    order = _generated_at_iso_list(isolated_index, "2026-05-13")
    # Expect descending — newest at top
    assert order == sorted(order, reverse=True), f"articles not desc: {order}"


def test_inserting_an_earlier_article_after_a_later_one_lands_at_bottom(isolated_index):
    """Order should be stable regardless of insertion sequence."""
    # Insert latest first
    html_archive.upsert_message("2026-05-13", "powerplay_1_1",
                                "2026-05-13T08:46:20-07:00", "PP1 body")
    # Then an earlier-timestamped one
    html_archive.upsert_message("2026-05-13", "morning",
                                "2026-05-13T00:01:13-07:00", "morning body")

    order = _generated_at_iso_list(isolated_index, "2026-05-13")
    assert order == sorted(order, reverse=True)
    # Newest at top
    assert order[0].startswith("2026-05-13T08:46")


def test_upserting_existing_article_keeps_descending_order(isolated_index):
    """When a message body is regenerated (same id, same generated_at), the
    article's position relative to its siblings should remain correct."""
    html_archive.upsert_message("2026-05-13", "morning",
                                "2026-05-13T00:01:13-07:00", "v1")
    html_archive.upsert_message("2026-05-13", "toss_1",
                                "2026-05-13T08:02:51-07:00", "toss v1")
    # Re-emit the toss with a fresher generated_at (as happens when delay
    # gates re-fire status updates)
    html_archive.upsert_message("2026-05-13", "toss_1",
                                "2026-05-13T08:02:51-07:00", "toss v2")

    order = _generated_at_iso_list(isolated_index, "2026-05-13")
    assert order == sorted(order, reverse=True)
