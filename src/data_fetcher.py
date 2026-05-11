"""Fetch IPL 2026 data from ESPN (primary) or Cricbuzz (fallback).

ESPN endpoints are undocumented but stable JSON. They return 403 from
some networks (cloud egress, datacenters) — fallback to Cricbuzz HTML
scraping with BeautifulSoup when that happens.

All cached files live in data/. TTLs per the spec:
  fixtures  : 24h
  squads    : 7d
  standings : 15m
  current   : 1m
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from .state import IST  # reuse zoneinfo

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ESPN_SERIES_ID = 1510719
CRICBUZZ_SERIES_ID = 8901  # IPL 2026 per cricbuzz URL slug

ESPN_BASE = "https://hs-consumer-api.espncricinfo.com/v1/pages/series"
ESPN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.espncricinfo.com/",
    "Origin": "https://www.espncricinfo.com",
}

TTL_SECONDS = {
    "fixtures": 24 * 3600,
    "squads": 7 * 24 * 3600,
    "standings": 15 * 60,
    "current": 60,
}


def _log(msg: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"\033[2m[{ts}]\033[0m {msg}", file=sys.stderr)


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.json"


def _read_cache(name: str, ttl: int) -> dict | None:
    p = _cache_path(name)
    if not p.exists():
        return None
    age = time.time() - p.stat().st_mtime
    if age > ttl:
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(name: str, payload: dict) -> None:
    with _cache_path(name).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _espn_get(path: str, params: dict) -> dict | None:
    url = f"{ESPN_BASE}/{path}"
    try:
        r = requests.get(url, headers=ESPN_HEADERS, params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
        _log(f"ESPN {path} returned {r.status_code}")
    except (requests.RequestException, ValueError) as e:
        _log(f"ESPN {path} error: {e}")
    return None


def _cricbuzz_html(path: str) -> BeautifulSoup | None:
    url = f"https://www.cricbuzz.com{path}"
    try:
        r = requests.get(url, headers={"User-Agent": ESPN_HEADERS["User-Agent"]}, timeout=10)
        if r.status_code == 200:
            return BeautifulSoup(r.text, "html.parser")
        _log(f"Cricbuzz {path} returned {r.status_code}")
    except requests.RequestException as e:
        _log(f"Cricbuzz {path} error: {e}")
    return None


# ─────────────────────────────────────────
# Public API: fixtures / standings / current match
# ─────────────────────────────────────────

def fetch_fixtures(force: bool = False) -> list[dict]:
    """Returns list of matches with id, teams (abbrev), scheduled_ist (HH:MM),
    date_ist (YYYY-MM-DD), status (scheduled/live/complete), result (str|None)."""
    if not force:
        cached = _read_cache("fixtures", TTL_SECONDS["fixtures"])
        if cached:
            return cached.get("matches", [])

    matches = _fixtures_from_espn() or _fixtures_from_cricbuzz() or []
    if matches:
        _write_cache("fixtures", {"fetched_at": datetime.now().isoformat(), "matches": matches})
    return matches


def fetch_standings(force: bool = False) -> list[dict]:
    """Returns list of {team, played, won, lost, points, nrr} ordered by position."""
    if not force:
        cached = _read_cache("standings", TTL_SECONDS["standings"])
        if cached:
            return cached.get("standings", [])

    standings = _standings_from_espn() or _standings_from_cricbuzz() or []
    if standings:
        _write_cache("standings", {"fetched_at": datetime.now().isoformat(), "standings": standings})
    return standings


def fetch_squads(force: bool = False) -> dict[str, dict]:
    """Returns {team_abbrev: {batters: [{name, runs}], bowlers: [{name, wickets}]}}.

    ESPN squads endpoint gives roster only; the stats endpoint gives
    top run-scorers/wicket-takers. We merge both for predictor input.
    """
    if not force:
        cached = _read_cache("squads", TTL_SECONDS["squads"])
        if cached:
            return cached.get("teams", {})

    teams = _squad_stats_from_espn() or {}
    if teams:
        _write_cache("squads", {"fetched_at": datetime.now().isoformat(), "teams": teams})
    return teams


def fetch_current_match(match_id: str) -> dict | None:
    """Returns latest status/result for a specific match (TTL 60s)."""
    cache_key = f"match_{match_id}"
    cached = _read_cache(cache_key, TTL_SECONDS["current"])
    if cached:
        return cached.get("match")

    match = _match_from_espn(match_id) or _match_from_cricbuzz(match_id)
    if match:
        _write_cache(cache_key, {"fetched_at": datetime.now().isoformat(), "match": match})
    return match


# ─────────────────────────────────────────
# ESPN parsers
# ─────────────────────────────────────────

def _normalize_team(name: str) -> str:
    """Map full name → 3/4 letter abbrev using data/teams.json aliases."""
    teams_path = DATA_DIR / "teams.json"
    if teams_path.exists():
        with teams_path.open() as f:
            t = json.load(f)
        aliases = t.get("aliases", {})
        if name in aliases:
            return aliases[name]
        # Tolerate trailing whitespace, alternate punctuation
        for k, v in aliases.items():
            if k.lower().strip() == name.lower().strip():
                return v
    return name


def _fixtures_from_espn() -> list[dict] | None:
    data = _espn_get("schedule", {"lang": "en", "seriesId": ESPN_SERIES_ID})
    if not data:
        return None
    matches: list[dict] = []
    # ESPN schedule shape: data["content"]["matches"] or data["matches"]
    raw_matches = (
        data.get("content", {}).get("matches")
        or data.get("matches")
        or []
    )
    for m in raw_matches:
        try:
            mid = str(m.get("objectId") or m.get("id"))
            teams = m.get("teams") or []
            if len(teams) < 2:
                continue
            t1 = _normalize_team(teams[0].get("team", {}).get("name", ""))
            t2 = _normalize_team(teams[1].get("team", {}).get("name", ""))
            start = m.get("startTime") or m.get("startDate")
            if not start:
                continue
            # ESPN gives ISO UTC; convert to IST
            dt = datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(IST)
            status_raw = (m.get("state") or "").lower()
            status = {
                "pre": "scheduled", "live": "live", "post": "complete",
            }.get(status_raw, status_raw or "scheduled")
            result = m.get("statusText") if status == "complete" else None
            matches.append({
                "id": mid,
                "teams": [t1, t2],
                "date_ist": dt.strftime("%Y-%m-%d"),
                "scheduled_ist": dt.strftime("%H:%M"),
                "status": status,
                "result": result,
            })
        except (KeyError, ValueError, TypeError) as e:
            _log(f"skipping malformed ESPN match: {e}")
    return matches or None


def _standings_from_espn() -> list[dict] | None:
    data = _espn_get("home", {"lang": "en", "seriesId": ESPN_SERIES_ID})
    if not data:
        return None
    # ESPN home payload includes "standings" or "table"
    table = data.get("standings") or data.get("table") or []
    if isinstance(table, dict):
        table = table.get("groups", [{}])[0].get("teamStats", [])
    out = []
    for row in table:
        try:
            team = _normalize_team(row.get("team", {}).get("name", "") or row.get("teamName", ""))
            out.append({
                "team": team,
                "played": int(row.get("played", row.get("matches", 0))),
                "won": int(row.get("won", row.get("wins", 0))),
                "lost": int(row.get("lost", row.get("losses", 0))),
                "points": int(row.get("points", 0)),
                "nrr": float(row.get("nrr", row.get("netRunRate", 0.0))),
            })
        except (ValueError, TypeError):
            continue
    return out or None


def _squad_stats_from_espn() -> dict[str, dict] | None:
    data = _espn_get("stats", {"lang": "en", "seriesId": ESPN_SERIES_ID, "trophyId": 117})
    if not data:
        return None
    teams: dict[str, dict] = {}
    # Stats payload typically has "mostRuns" and "mostWickets" arrays
    for player in data.get("mostRuns", []) or []:
        try:
            team = _normalize_team(player.get("team", {}).get("name", ""))
            teams.setdefault(team, {"batters": [], "bowlers": []})
            teams[team]["batters"].append({
                "name": player.get("name") or player.get("playerName"),
                "runs": int(player.get("runs", 0)),
            })
        except (ValueError, TypeError):
            continue
    for player in data.get("mostWickets", []) or []:
        try:
            team = _normalize_team(player.get("team", {}).get("name", ""))
            teams.setdefault(team, {"batters": [], "bowlers": []})
            teams[team]["bowlers"].append({
                "name": player.get("name") or player.get("playerName"),
                "wickets": int(player.get("wickets", 0)),
            })
        except (ValueError, TypeError):
            continue
    # Sort each team's lists desc and trim to top 3
    for team in teams.values():
        team["batters"] = sorted(team["batters"], key=lambda p: p["runs"], reverse=True)[:3]
        team["bowlers"] = sorted(team["bowlers"], key=lambda p: p["wickets"], reverse=True)[:3]
    return teams or None


def _match_from_espn(match_id: str) -> dict | None:
    # ESPN match-detail endpoint
    url = f"https://hs-consumer-api.espncricinfo.com/v1/pages/match/home"
    try:
        r = requests.get(url, headers=ESPN_HEADERS, params={"lang": "en", "seriesId": ESPN_SERIES_ID, "matchId": match_id}, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        m = data.get("match") or {}
        state = (m.get("state") or "").lower()
        status = {"pre": "scheduled", "live": "live", "post": "complete"}.get(state, state)
        teams = m.get("teams", [])
        winner = None
        if status == "complete":
            for t in teams:
                if t.get("isWinner"):
                    winner = _normalize_team(t.get("team", {}).get("name", ""))
                    break
        return {
            "id": match_id,
            "status": status,
            "result": m.get("statusText"),
            "winner": winner,
        }
    except (requests.RequestException, ValueError, KeyError):
        return None


# ─────────────────────────────────────────
# Cricbuzz fallback (HTML scrape)
# ─────────────────────────────────────────

def _fixtures_from_cricbuzz() -> list[dict] | None:
    soup = _cricbuzz_html(f"/cricket-series/{CRICBUZZ_SERIES_ID}/indian-premier-league-2026/matches")
    if not soup:
        return None
    matches: list[dict] = []
    # Cricbuzz match rows: .cb-srs-mtchs-tm or similar; structure changes — be defensive
    for row in soup.select("div.cb-srs-mtchs-tm, div.cb-col.cb-col-100.cb-series-matches"):
        try:
            link = row.find("a", href=True)
            if not link:
                continue
            href = link["href"]  # /live-cricket-scores/<mid>/...
            mid = href.strip("/").split("/")[1] if "/" in href else ""
            text = link.get_text(" ", strip=True)
            # Format: "Team A vs Team B, 1st Match"
            teams_part = text.split(",")[0]
            if " vs " not in teams_part:
                continue
            t1_raw, t2_raw = [s.strip() for s in teams_part.split(" vs ", 1)]
            t1 = _normalize_team(t1_raw)
            t2 = _normalize_team(t2_raw)
            # Date/time often in a sibling div
            time_div = row.find("div", class_="schedule-date") or row.find("span", class_="schedule-date")
            scheduled_iso = time_div.get("timestamp") if time_div and time_div.has_attr("timestamp") else None
            date_ist = ""
            scheduled_ist = ""
            if scheduled_iso:
                try:
                    dt = datetime.fromtimestamp(int(scheduled_iso) / 1000, tz=IST)
                    date_ist = dt.strftime("%Y-%m-%d")
                    scheduled_ist = dt.strftime("%H:%M")
                except (ValueError, TypeError):
                    pass
            matches.append({
                "id": mid,
                "teams": [t1, t2],
                "date_ist": date_ist,
                "scheduled_ist": scheduled_ist,
                "status": "scheduled",
                "result": None,
            })
        except (KeyError, AttributeError, ValueError) as e:
            _log(f"skipping cricbuzz row: {e}")
    return matches or None


def _standings_from_cricbuzz() -> list[dict] | None:
    soup = _cricbuzz_html(f"/cricket-series/{CRICBUZZ_SERIES_ID}/indian-premier-league-2026/points-table")
    if not soup:
        return None
    rows = soup.select("table.cb-srs-pnts tr")
    out = []
    for tr in rows[1:]:  # skip header
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) < 7:
            continue
        try:
            out.append({
                "team": _normalize_team(cells[0]),
                "played": int(cells[1]),
                "won": int(cells[2]),
                "lost": int(cells[3]),
                "points": int(cells[6]) if cells[6].isdigit() else 0,
                "nrr": float(cells[7]) if len(cells) > 7 and cells[7].replace("-", "").replace(".", "").isdigit() else 0.0,
            })
        except (ValueError, IndexError):
            continue
    return out or None


def _match_from_cricbuzz(match_id: str) -> dict | None:
    soup = _cricbuzz_html(f"/live-cricket-scores/{match_id}")
    if not soup:
        return None
    status_el = soup.find("div", class_="cb-text-complete") or soup.find("div", class_="cb-text-inprogress")
    if not status_el:
        return None
    status = "complete" if "cb-text-complete" in (status_el.get("class") or []) else "live"
    return {
        "id": match_id,
        "status": status,
        "result": status_el.get_text(strip=True),
        "winner": None,  # would need extra parsing
    }


if __name__ == "__main__":
    # Quick manual test
    _log("Fetching fixtures...")
    fx = fetch_fixtures(force=True)
    _log(f"Got {len(fx)} matches")
    if fx:
        print(json.dumps(fx[:3], indent=2))
    _log("Fetching standings...")
    st = fetch_standings(force=True)
    _log(f"Got {len(st)} teams in table")
    if st:
        print(json.dumps(st[:4], indent=2))
