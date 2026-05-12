"""v6 Phase B1 — historical weather backfill from open-meteo archive API.

For every match in 2008-2025 with a known venue, fetch the hourly weather at
match-local 19:00 (IPL evening starts ~19:30 local time). The open-meteo
archive endpoint is free, key-less, and rate-limited to 10K req/day. We
throttle to ~1 req/sec to stay polite.

Responses are cached as JSON under `ml/data/historical/v6/raw/weather_<mid>.json`
(gitignored). Re-runs reuse the cache and only hit the network for misses.

Output columns: match_id, temp_c, humidity_pct, dewpoint_c, wind_kmh, precipitation_mm.
NaN where the API fails or the venue has no geocode. Failures are recorded to
PROGRESS.md (appended block).

CLI: `python -m ml.src.v6.build_b1 [--limit N] [--sleep 1.0]`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from ml.src.v6._io import HIST, V6_DIR, load_matches

OUTPUT = V6_DIR / "weather.parquet"
RAW_DIR = V6_DIR / "raw"
GEOCODES_PATH = V6_DIR / "venue_geocodes.json"

API = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "temperature_2m",
    "relativehumidity_2m",
    "dewpoint_2m",
    "wind_speed_10m",
    "precipitation",
]
TARGET_HOUR = 19  # 19:00 local; IPL evening matches start ~19:30


def load_geocodes() -> dict[str, dict]:
    with open(GEOCODES_PATH) as f:
        return json.load(f)


def cache_path(match_id: str) -> pathlib.Path:
    return RAW_DIR / f"weather_{match_id}.json"


def fetch_one(match_id: str, lat: float, lon: float, date_str: str, tz: str,
              session: requests.Session, timeout: float = 30.0) -> dict:
    """Fetch (or load from cache) the open-meteo response for one (match, day).

    Returns the parsed JSON. Caches on disk on success.
    """
    cp = cache_path(match_id)
    if cp.exists():
        try:
            with open(cp) as f:
                data = json.load(f)
            if isinstance(data, dict) and "hourly" in data:
                return data
        except (json.JSONDecodeError, OSError):
            pass  # fall through to refetch

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": tz,
    }
    r = session.get(API, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cp.with_suffix(cp.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, cp)
    return data


def extract_row(data: dict, hour: int = TARGET_HOUR) -> dict:
    """Pull the values at the target local hour out of an open-meteo response."""
    h = data.get("hourly") or {}
    times = h.get("time") or []
    if not times:
        return {k: math.nan for k in ["temp_c", "humidity_pct", "dewpoint_c",
                                       "wind_kmh", "precipitation_mm"]}

    # times are local-tz ISO strings; find the row at hour=19
    idx = None
    for i, t in enumerate(times):
        # format "YYYY-MM-DDTHH:MM"
        try:
            hh = int(t[11:13])
        except (ValueError, IndexError):
            continue
        if hh == hour:
            idx = i
            break
    if idx is None:
        idx = min(hour, len(times) - 1)  # graceful fallback

    def at(name):
        col = h.get(name)
        if not col or idx >= len(col):
            return math.nan
        v = col[idx]
        return math.nan if v is None else float(v)

    return {
        "temp_c": at("temperature_2m"),
        "humidity_pct": at("relativehumidity_2m"),
        "dewpoint_c": at("dewpoint_2m"),
        "wind_kmh": at("wind_speed_10m"),
        "precipitation_mm": at("precipitation"),
    }


def write_atomic_weather(df: pd.DataFrame) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    schema = pa.schema([
        ("match_id", pa.string()),
        ("temp_c", pa.float64()),
        ("humidity_pct", pa.float64()),
        ("dewpoint_c", pa.float64()),
        ("wind_kmh", pa.float64()),
        ("precipitation_mm", pa.float64()),
    ])
    table = pa.Table.from_pandas(df[["match_id", "temp_c", "humidity_pct",
                                      "dewpoint_c", "wind_kmh", "precipitation_mm"]],
                                  schema=schema, preserve_index=False)
    pq.write_table(table, tmp)
    os.replace(tmp, OUTPUT)


def run(limit: int | None = None, sleep_sec: float = 1.0) -> dict:
    matches = load_matches()
    matches = matches.dropna(subset=["venue"]).copy()
    matches["match_id"] = matches["match_id"].astype(str)
    matches["date_str"] = pd.to_datetime(matches["date"]).dt.strftime("%Y-%m-%d")

    geocodes = load_geocodes()

    if limit is not None:
        matches = matches.head(limit)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    rows = []
    failures: list[tuple[str, str, str]] = []  # (match_id, venue, reason)
    no_geo: list[tuple[str, str]] = []

    total = len(matches)
    last_log = time.time()
    for i, (_, m) in enumerate(matches.iterrows(), start=1):
        mid = m["match_id"]
        venue = m["venue"]
        date_str = m["date_str"]
        gc = geocodes.get(venue)
        if gc is None:
            no_geo.append((mid, venue))
            rows.append({"match_id": mid, **{k: math.nan for k in [
                "temp_c", "humidity_pct", "dewpoint_c", "wind_kmh", "precipitation_mm"]}})
            continue

        cached = cache_path(mid).exists()
        try:
            data = fetch_one(mid, gc["lat"], gc["lon"], date_str, gc["tz"], session)
            extracted = extract_row(data, hour=TARGET_HOUR)
        except (requests.RequestException, ValueError) as e:
            failures.append((mid, venue, str(e)[:200]))
            extracted = {k: math.nan for k in ["temp_c", "humidity_pct", "dewpoint_c",
                                                "wind_kmh", "precipitation_mm"]}

        rows.append({"match_id": mid, **extracted})

        # throttle only when we actually hit the network
        if not cached:
            time.sleep(sleep_sec)

        now = time.time()
        if now - last_log >= 10 or i == total:
            cache_hits = sum(1 for r in rows if not math.isnan(r.get("temp_c", math.nan)))
            print(f"  [{i}/{total}] ok={cache_hits} fail={len(failures)} no_geo={len(no_geo)}", flush=True)
            last_log = now

    df = pd.DataFrame(rows)
    write_atomic_weather(df)

    ok_count = df["temp_c"].notna().sum()
    return {
        "rows": len(df),
        "ok": int(ok_count),
        "failures": failures,
        "no_geo": no_geo,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N matches (debug)")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="seconds between live API calls (default 1.0)")
    args = ap.parse_args(argv)

    summary = run(limit=args.limit, sleep_sec=args.sleep)
    print(f"\nwrote {OUTPUT}")
    print(f"  rows: {summary['rows']}")
    print(f"  ok:   {summary['ok']}  ({summary['ok'] / max(summary['rows'],1):.1%})")
    if summary["failures"]:
        print(f"  failures: {len(summary['failures'])}")
        for mid, venue, reason in summary["failures"][:10]:
            print(f"    {mid}  {venue!r}  {reason}")
    if summary["no_geo"]:
        print(f"  no_geo: {len(summary['no_geo'])} (should be 0)")
        for mid, venue in summary["no_geo"][:10]:
            print(f"    {mid}  {venue!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
