"""Fetch IPL 2026 data — tiered source strategy.

Tiers (in order; first that yields data wins):
  1. IPLT20 official feed     (ipl-stats-sports-mechanic.s3.ap-south-1.amazonaws.com)
  2. ESPN Cricinfo JSON API   (hs-consumer-api.espncricinfo.com)
  3. CricAPI                  (api.cricapi.com — requires CRICAPI_KEY env var)
  4. Cricbuzz HTML scrape     (floor — limited match info, no standings)

The IPLT20 S3-backed feed is what iplt20.com itself uses. It's public, no key,
no auth, no rate limit, and contains: full schedule, points table (with last-5
form), top run scorers, most wickets. CompetitionID 284 = IPL 2026.

Cache TTLs (per spec):
  fixtures  : 24h     squads    : 7d
  standings : 15m     current   : 1m
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .state import IST  # reuse zoneinfo

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Primary: iplt20.com official feed
IPLT20_BASE = "https://ipl-stats-sports-mechanic.s3.ap-south-1.amazonaws.com/ipl/feeds"
IPLT20_COMP_ID = 284  # IPL 2026 (from competition.js)

ESPN_SERIES_ID = 1510719
CRICBUZZ_SERIES_ID = 8901
CRICAPI_SERIES_ID_FALLBACK = "ipl-2026"

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

    matches = (
        _fixtures_from_iplt20()
        or _fixtures_from_espn()
        or _fixtures_from_cricapi()
        or _fixtures_from_cricbuzz()
        or []
    )
    if matches:
        _write_cache("fixtures", {"fetched_at": datetime.now().isoformat(), "matches": matches})
    return matches


def fetch_standings(force: bool = False) -> list[dict]:
    """Returns list of {team, played, won, lost, points, nrr} ordered by position.

    Cricbuzz standings aren't HTML-scrapable (loaded client-side), so the
    fallback is CricAPI only. If neither works, returns []."""
    if not force:
        cached = _read_cache("standings", TTL_SECONDS["standings"])
        if cached:
            return cached.get("standings", [])

    standings = (
        _standings_from_iplt20()
        or _standings_from_espn()
        or _standings_from_cricapi()
        or []
    )
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

    teams = _squad_stats_from_iplt20() or _squad_stats_from_espn() or {}
    if teams:
        _write_cache("squads", {"fetched_at": datetime.now().isoformat(), "teams": teams})
    return teams


def fetch_current_match(match_id: str) -> dict | None:
    """Returns latest status/result for a specific match (TTL 60s)."""
    cache_key = f"match_{match_id}"
    cached = _read_cache(cache_key, TTL_SECONDS["current"])
    if cached:
        return cached.get("match")

    match = (
        _match_from_iplt20(match_id)
        or _match_from_espn(match_id)
        or _match_from_cricbuzz(match_id)
    )
    if match:
        _write_cache(cache_key, {"fetched_at": datetime.now().isoformat(), "match": match})
    return match


# ─────────────────────────────────────────
# IPLT20 official feed (primary)
# ─────────────────────────────────────────

_JSONP_RE = re.compile(r"^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\(\s*(.+?)\s*\)\s*;?\s*$", re.DOTALL)


def _iplt20_fetch(path: str) -> dict | None:
    url = f"{IPLT20_BASE}{path}"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if r.status_code != 200:
            _log(f"iplt20 {path} returned {r.status_code}")
            return None
        text = r.text
        m = _JSONP_RE.match(text)
        payload = m.group(1) if m else text
        return json.loads(payload)
    except (requests.RequestException, ValueError) as e:
        _log(f"iplt20 {path} error: {e}")
        return None


_IPLT20_STATUS_MAP = {
    "post": "complete",
    "live": "live",
    "in progress": "live",
    "pre": "scheduled",
    "upcoming": "scheduled",
    "scheduled": "scheduled",
}


def _parse_iplt20_match(m: dict) -> dict | None:
    try:
        mid = str(m.get("MatchID"))
        t1 = m.get("FirstBattingTeamCode") or _normalize_team(m.get("FirstBattingTeamName", ""))
        t2 = m.get("SecondBattingTeamCode") or _normalize_team(m.get("SecondBattingTeamName", ""))
        date_ist = m.get("MatchDate") or m.get("MatchDateNew")
        # MatchDate is "YYYY-MM-DD"; MatchTime is "HH:MM" (24h, IST)
        time_ist = m.get("MatchTime") or "00:00"
        if not date_ist or "-" not in date_ist:
            # Try parsing MatchDateNew "11 May 2026"
            try:
                date_ist = datetime.strptime(m.get("MatchDateNew", ""), "%d %b %Y").strftime("%Y-%m-%d")
            except ValueError:
                return None
        status_raw = (m.get("MatchStatus") or "").lower().strip()
        status = _IPLT20_STATUS_MAP.get(status_raw, "scheduled" if status_raw else "scheduled")
        # Winner
        winner = None
        result_text = None
        if status == "complete":
            wt_id = m.get("WinningTeamID")
            if wt_id and str(wt_id) == str(m.get("FirstBattingTeamID")):
                winner = t1
            elif wt_id and str(wt_id) == str(m.get("SecondBattingTeamID")):
                winner = t2
            result_text = (m.get("Commentss") or "").strip() or None
        # Richer fields for the predictor
        home_team = None
        ht_id = m.get("HomeTeamID")
        if ht_id:
            if str(ht_id) == str(m.get("FirstBattingTeamID")):
                home_team = t1
            elif str(ht_id) == str(m.get("SecondBattingTeamID")):
                home_team = t2
            else:
                home_team = _normalize_team(m.get("HomeTeamName") or "")

        toss_winner = None
        toss_team = (m.get("TossTeam") or "").strip()
        if toss_team:
            toss_winner = _normalize_team(toss_team)
        toss_decision = None
        toss_text = (m.get("TossText") or "").lower()
        if "field" in toss_text or "bowl" in toss_text:
            toss_decision = "field"
        elif "bat" in toss_text:
            toss_decision = "bat"

        def _to_int(v):
            try:
                s = str(v).strip()
                return int(s) if s and s.lstrip("-").isdigit() else 0
            except (ValueError, TypeError):
                return 0

        mom = None
        if m.get("MOMPlayerId") or m.get("MOM"):
            mom = {
                "player_id": str(m.get("MOMPlayerId") or ""),
                "name": (m.get("MOM") or "").strip(),
                "runs":   _to_int(m.get("MOMRuns")),
                "balls":  _to_int(m.get("MOMBalls")),
                "wickets":_to_int(m.get("MOMWicket")),
                "rc":     _to_int(m.get("MOMRC")),
            }

        return {
            "id": mid,
            "teams": [_normalize_team(t1), _normalize_team(t2)],
            "date_ist": date_ist,
            "scheduled_ist": time_ist,
            "status": status,
            "result": result_text,
            "winner": winner,
            "first_batting": _normalize_team(t1),
            "second_batting": _normalize_team(t2),
            "home_team": home_team,
            "venue_id": str(m.get("GroundID") or ""),
            "venue_name": (m.get("GroundName") or "").strip(),
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "mom": mom,
        }
    except (KeyError, ValueError, TypeError) as e:
        _log(f"iplt20: skipping match: {e}")
        return None


def _fixtures_from_iplt20() -> list[dict] | None:
    data = _iplt20_fetch(f"/{IPLT20_COMP_ID}-matchschedule.js")
    if not data:
        return None
    raw = data.get("Matchsummary") or []
    matches = [m for m in (_parse_iplt20_match(r) for r in raw) if m]
    if matches:
        _log(f"iplt20: parsed {len(matches)} matches from official feed")
    return matches or None


def _standings_from_iplt20() -> list[dict] | None:
    data = _iplt20_fetch(f"/stats/{IPLT20_COMP_ID}-groupstandings.js")
    if not data:
        return None
    rows = data.get("points") or []
    out: list[dict] = []
    for r in rows:
        try:
            # ForTeams/AgainstTeam look like "2026/195.1" — runs scored / overs
            def _split_for(s):
                if not s or "/" not in s:
                    return 0.0, 0.0
                try:
                    runs, overs = s.split("/", 1)
                    return float(runs), float(overs)
                except (ValueError, TypeError):
                    return 0.0, 0.0
            for_runs, for_overs = _split_for(r.get("ForTeams"))
            against_runs, against_overs = _split_for(r.get("AgainstTeam"))

            out.append({
                "team": r.get("TeamCode") or _normalize_team(r.get("TeamName", "")),
                "played": int(r.get("Matches", 0)),
                "won": int(r.get("Wins", 0)),
                "lost": int(r.get("Loss", 0)),
                "points": int(r.get("Points", 0)),
                "nrr": float(r.get("NetRunRate", 0.0) or 0.0),
                "performance": r.get("Performance") or "",  # "W,W,L,L,W"
                "order": int(r.get("OrderNo", 0) or 0),
                "is_qualified": bool(r.get("IsQualified")),
                "for_runs": for_runs,
                "for_overs": for_overs,
                "against_runs": against_runs,
                "against_overs": against_overs,
            })
        except (ValueError, TypeError):
            continue
    out.sort(key=lambda r: r.get("order") or 99)
    return out or None


def _squad_stats_from_iplt20() -> dict[str, dict] | None:
    teams: dict[str, dict] = {}
    runs = _iplt20_fetch(f"/stats/{IPLT20_COMP_ID}-toprunsscorers.js")
    if runs:
        for p in runs.get("toprunsscorers", []) or []:
            try:
                code = p.get("TeamCode")
                if not code:
                    continue
                teams.setdefault(code, {"batters": [], "bowlers": []})
                teams[code]["batters"].append({
                    "name": p.get("StrikerName"),
                    "runs": int(p.get("TotalRuns", 0)),
                })
            except (ValueError, TypeError):
                continue
    wickets = _iplt20_fetch(f"/stats/{IPLT20_COMP_ID}-mostwickets.js")
    if wickets:
        # Find the wicket count field — try a few likely names
        rows = list(wickets.values())[0] if wickets else []
        for p in rows or []:
            try:
                code = p.get("TeamCode")
                if not code:
                    continue
                wkts = p.get("Wickets") or p.get("TotalWickets") or p.get("WicketsTaken") or p.get("wickets")
                if wkts is None:
                    continue
                teams.setdefault(code, {"batters": [], "bowlers": []})
                teams[code]["bowlers"].append({
                    "name": p.get("BowlerName"),
                    "wickets": int(wkts),
                })
            except (ValueError, TypeError):
                continue
    # Top 3 batters / bowlers per team
    for code, t in teams.items():
        t["batters"] = sorted(t["batters"], key=lambda b: b["runs"], reverse=True)[:3]
        t["bowlers"] = sorted(t["bowlers"], key=lambda b: b["wickets"], reverse=True)[:3]
    return teams or None


def _match_from_iplt20(match_id: str) -> dict | None:
    # The schedule feed already carries live state. Re-fetch (cached 60s upstream).
    data = _iplt20_fetch(f"/{IPLT20_COMP_ID}-matchschedule.js")
    if not data:
        return None
    for raw in data.get("Matchsummary", []) or []:
        if str(raw.get("MatchID")) == str(match_id):
            parsed = _parse_iplt20_match(raw)
            if parsed:
                return {
                    "id": match_id,
                    "status": parsed["status"],
                    "result": parsed["result"],
                    "winner": parsed["winner"],
                }
    return None


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
# CricAPI tier — requires CRICAPI_KEY env var (free tier: ~100 calls/day)
# Docs: https://cricapi.com/howto
# ─────────────────────────────────────────

def _cricapi_key() -> str | None:
    key = os.environ.get("CRICAPI_KEY", "").strip()
    return key or None


def _cricapi_get(endpoint: str, params: dict | None = None) -> dict | None:
    key = _cricapi_key()
    if not key:
        return None
    p = {"apikey": key, **(params or {})}
    try:
        r = requests.get(f"https://api.cricapi.com/v1/{endpoint}", params=p, timeout=10)
        if r.status_code != 200:
            _log(f"cricapi {endpoint} returned {r.status_code}")
            return None
        data = r.json()
        if data.get("status") != "success":
            _log(f"cricapi {endpoint}: {data.get('reason', 'unknown error')}")
            return None
        return data
    except (requests.RequestException, ValueError) as e:
        _log(f"cricapi {endpoint} error: {e}")
        return None


def _resolve_cricapi_series_id() -> str | None:
    """Find the live IPL 2026 series ID via CricAPI's series search."""
    data = _cricapi_get("series", {"search": "Indian Premier League 2026"})
    if not data:
        return None
    for s in data.get("data", []):
        name = (s.get("name") or "").lower()
        if "indian premier league" in name and "2026" in name:
            return s.get("id")
    return None


def _fixtures_from_cricapi() -> list[dict] | None:
    sid = _resolve_cricapi_series_id()
    if not sid:
        return None
    data = _cricapi_get("series_info", {"id": sid})
    if not data:
        return None
    info = data.get("data", {})
    matches: list[dict] = []
    for m in info.get("matchList", []) or info.get("matches", []) or []:
        try:
            mid = str(m.get("id"))
            teams = m.get("teams") or []
            if len(teams) < 2:
                continue
            t1 = _normalize_team(teams[0])
            t2 = _normalize_team(teams[1])
            date_str = m.get("dateTimeGMT") or m.get("date")
            if not date_str:
                continue
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00")).astimezone(IST)
            status_raw = (m.get("status") or "").lower()
            if any(k in status_raw for k in ("won", "tie", "no result", "abandoned")):
                status = "complete"
            elif m.get("matchStarted") and not m.get("matchEnded"):
                status = "live"
            else:
                status = "scheduled"
            winner = _normalize_team(m.get("matchWinner", "")) if status == "complete" else None
            matches.append({
                "id": mid,
                "teams": [t1, t2],
                "date_ist": dt.strftime("%Y-%m-%d"),
                "scheduled_ist": dt.strftime("%H:%M"),
                "status": status,
                "result": m.get("status") if status == "complete" else None,
                "winner": winner,
            })
        except (ValueError, TypeError, KeyError) as e:
            _log(f"cricapi: skipping malformed match: {e}")
    return matches or None


def _standings_from_cricapi() -> list[dict] | None:
    sid = _resolve_cricapi_series_id()
    if not sid:
        return None
    data = _cricapi_get("series_points", {"id": sid})
    if not data:
        return None
    out: list[dict] = []
    for row in data.get("data", []) or []:
        try:
            out.append({
                "team": _normalize_team(row.get("teamname") or row.get("team", "")),
                "played": int(row.get("matches", 0)),
                "won": int(row.get("wins", row.get("won", 0))),
                "lost": int(row.get("loss", row.get("lost", 0))),
                "points": int(row.get("points", 0)),
                "nrr": float(row.get("nrr", 0.0)),
            })
        except (ValueError, TypeError):
            continue
    return out or None


# ─────────────────────────────────────────
# Cricbuzz tier — HTML scrape (floor)
# Limited to the 4 matches visible on the schedule page (recent + upcoming).
# Standings can't be scraped from HTML — Cricbuzz renders them client-side.
# ─────────────────────────────────────────

_SLUG_RE = re.compile(r"/live-cricket-scores/(\d+)/([a-z0-9-]+)")
_ANCHOR_TXT_RE = re.compile(r"^([A-Z]{2,4})vs([A-Z]{2,4})-(.+)$")
_DATE_RE = re.compile(
    r"([A-Z][a-z]+),\s*([A-Z][a-z]+)\s+(\d{1,2}),\s*(\d{4}),\s*(\d{1,2}:\d{2})\s*(AM|PM)\s*IST",
    re.IGNORECASE,
)


def _parse_cricbuzz_anchor(href: str, text: str) -> dict | None:
    m = _SLUG_RE.search(href)
    if not m:
        return None
    mid, slug = m.group(1), m.group(2)
    if "indian-premier-league-2026" not in slug:
        return None
    text_m = _ANCHOR_TXT_RE.match(text.strip())
    if not text_m:
        # Fallback: parse teams from slug "csk-vs-mi-12th-match-..."
        slug_m = re.match(r"^([a-z]+)-vs-([a-z]+)-", slug)
        if not slug_m:
            return None
        t1, t2 = slug_m.group(1).upper(), slug_m.group(2).upper()
        status_hint = ""
    else:
        t1, t2, status_hint = text_m.group(1), text_m.group(2), text_m.group(3).strip()

    status_low = status_hint.lower()
    if "won" in status_low or "tie" in status_low:
        status = "complete"
        winner = _normalize_team(status_low.split(" won")[0].strip().upper()) if " won" in status_low else None
    elif "live" in status_low or "inning" in status_low:
        status = "live"
        winner = None
    else:
        status = "scheduled"
        winner = None

    return {
        "id": mid,
        "teams": [_normalize_team(t1), _normalize_team(t2)],
        "slug": slug,
        "status": status,
        "winner": winner,
        "result": status_hint if status == "complete" else None,
        "date_ist": "",
        "scheduled_ist": "",
    }


def _enrich_cricbuzz_match(match: dict) -> dict:
    """Fetch the per-match page to fill date/time. Cached on disk per match id."""
    cache_key = f"cb_match_{match['id']}"
    cached = _read_cache(cache_key, TTL_SECONDS["current"])
    if cached and cached.get("match", {}).get("date_ist"):
        match.update({k: v for k, v in cached["match"].items() if v})
        return match
    soup = _cricbuzz_html(f"/live-cricket-scores/{match['id']}/{match.get('slug','')}")
    if not soup:
        return match
    page_text = soup.get_text(" ", strip=True)
    d = _DATE_RE.search(page_text)
    if d:
        try:
            dt_str = f"{d.group(2)} {d.group(3)} {d.group(4)} {d.group(5)} {d.group(6)}"
            dt = datetime.strptime(dt_str, "%B %d %Y %I:%M %p").replace(tzinfo=IST)
            match["date_ist"] = dt.strftime("%Y-%m-%d")
            match["scheduled_ist"] = dt.strftime("%H:%M")
        except ValueError:
            pass
    _write_cache(cache_key, {"fetched_at": datetime.now().isoformat(), "match": match})
    return match


def _fixtures_from_cricbuzz() -> list[dict] | None:
    soup = _cricbuzz_html(f"/cricket-series/{CRICBUZZ_SERIES_ID}/indian-premier-league-2026/matches")
    if not soup:
        return None
    seen: set[str] = set()
    matches: list[dict] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/live-cricket-scores/" not in href or "indian-premier-league-2026" not in href:
            continue
        parsed = _parse_cricbuzz_anchor(href, a.get_text(strip=True))
        if not parsed or parsed["id"] in seen:
            continue
        seen.add(parsed["id"])
        matches.append(_enrich_cricbuzz_match(parsed))
    if not matches:
        _log("cricbuzz: no IPL 2026 match links found on schedule page")
        return None
    _log(f"cricbuzz: parsed {len(matches)} matches from schedule page")
    return matches


def _match_from_cricbuzz(match_id: str) -> dict | None:
    """Fetch a single match's live status. We don't have the slug here so try
    the canonical short URL; Cricbuzz redirects to the slugged URL."""
    soup = _cricbuzz_html(f"/live-cricket-scores/{match_id}")
    if not soup:
        return None
    page_text = soup.get_text(" ", strip=True)
    # Status appears as e.g. "<TEAM> won by 12 runs" or "Match starts at..."
    won_m = re.search(r"([A-Z][a-zA-Z ]+?)\s+won by\s+(\d+\s+\w+)", page_text)
    if won_m:
        return {
            "id": match_id,
            "status": "complete",
            "result": won_m.group(0),
            "winner": _normalize_team(won_m.group(1).strip()),
        }
    if "Match starts at" in page_text:
        return {"id": match_id, "status": "scheduled", "result": None, "winner": None}
    # Possibly live / in-progress
    if re.search(r"\d+/\d+", page_text):
        return {"id": match_id, "status": "live", "result": None, "winner": None}
    return None


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
