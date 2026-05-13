"""Tests for html_archive.resort_all_days — a one-shot helper that re-orders
every existing day section's articles in descending (newest-first) order.

Used to heal legacy ascending day sections that pre-date the upsert_message
flip; will also serve if we ever need to re-flip in the future.
"""
from __future__ import annotations

import pytest

from src import html_archive


@pytest.fixture
def isolated_index(tmp_path, monkeypatch):
    idx = tmp_path / "index.html"
    monkeypatch.setattr(html_archive, "INDEX", idx)
    monkeypatch.setattr(html_archive, "DOCS_DIR", tmp_path)
    return idx


def _gens(idx_path, date_iso):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(idx_path.read_text(), "html.parser")
    details = soup.find("details", attrs={"data-day": date_iso})
    assert details is not None
    return [a.get("data-generated", "") for a in details.find_all("article", recursive=False)]


def _seed_ascending(date_iso):
    """Force a day section into ascending order, the legacy layout."""
    html_archive.upsert_message(date_iso, "morning",
                                f"{date_iso}T00:01:13-07:00", "morning")
    html_archive.upsert_message(date_iso, "toss_1",
                                f"{date_iso}T08:02:00-07:00", "toss")
    html_archive.upsert_message(date_iso, "powerplay_1_1",
                                f"{date_iso}T08:46:00-07:00", "PP1")
    # Undo the descending sort that upsert_message now does, by manually
    # rewriting the file with siblings in ascending order.
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_archive.INDEX.read_text(), "html.parser")
    details = soup.find("details", attrs={"data-day": date_iso})
    arts = list(details.find_all("article", recursive=False))
    arts.sort(key=lambda a: a.get("data-generated", "") or "")
    for a in arts:
        a.extract()
    for a in arts:
        details.append(a)
    html_archive.INDEX.write_text(str(soup), encoding="utf-8")


def test_resort_all_days_flips_ascending_to_descending(isolated_index):
    _seed_ascending("2026-05-11")
    asc = _gens(isolated_index, "2026-05-11")
    assert asc == sorted(asc), "seeding should produce ascending order"

    html_archive.resort_all_days()

    desc = _gens(isolated_index, "2026-05-11")
    assert desc == sorted(desc, reverse=True)


def test_resort_all_days_handles_multiple_days(isolated_index):
    _seed_ascending("2026-05-11")
    _seed_ascending("2026-05-12")

    html_archive.resort_all_days()

    for d in ("2026-05-11", "2026-05-12"):
        order = _gens(isolated_index, d)
        assert order == sorted(order, reverse=True), f"{d} not desc: {order}"


def test_resort_all_days_idempotent(isolated_index):
    _seed_ascending("2026-05-11")
    html_archive.resort_all_days()
    first = _gens(isolated_index, "2026-05-11")
    html_archive.resort_all_days()
    second = _gens(isolated_index, "2026-05-11")
    assert first == second


def test_resort_all_days_noop_when_no_index(tmp_path, monkeypatch):
    """If docs/index.html doesn't exist, the function should not crash."""
    idx = tmp_path / "index.html"
    monkeypatch.setattr(html_archive, "INDEX", idx)
    monkeypatch.setattr(html_archive, "DOCS_DIR", tmp_path)
    # Doesn't raise
    html_archive.resort_all_days()
