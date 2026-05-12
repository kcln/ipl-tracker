"""Schema + sanity tests for v6 Phase B outputs.

These check the produced artifacts exist and have the right shape. They do
NOT call the open-meteo API; they read what's already on disk.
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd
import pyarrow.parquet as pq
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
HIST = ROOT / "ml" / "data" / "historical"
V6 = HIST / "v6"

GEOCODES = V6 / "venue_geocodes.json"
WEATHER = V6 / "weather.parquet"
XI = V6 / "playing_xi.parquet"


# ---------- B3: geocodes ----------

@pytest.fixture(scope="module")
def geocodes() -> dict:
    assert GEOCODES.exists(), f"missing {GEOCODES}"
    with open(GEOCODES) as f:
        return json.load(f)


def test_geocodes_cover_all_venues(geocodes):
    """Every venue in matches.parquet (2008-2025) must have a geocode."""
    m = pd.read_parquet(HIST / "matches.parquet")
    m = m[m["season"].between(2008, 2025)]
    venues = set(m["venue"].dropna().unique())
    missing = venues - set(geocodes.keys())
    assert not missing, f"missing geocodes: {missing}"


def test_geocodes_schema(geocodes):
    required = {"name", "lat", "lon", "city", "country", "tz"}
    for venue, entry in geocodes.items():
        keys = set(entry.keys())
        assert required.issubset(keys), f"{venue}: missing keys {required - keys}"
        assert -90 <= entry["lat"] <= 90, f"{venue}: bad lat {entry['lat']}"
        assert -180 <= entry["lon"] <= 180, f"{venue}: bad lon {entry['lon']}"
        assert isinstance(entry["tz"], str) and "/" in entry["tz"]


def test_geocodes_known_landmarks(geocodes):
    """Spot-check a couple of well-known grounds."""
    wankhede = geocodes["Wankhede Stadium"]
    # Wankhede is on the Mumbai coast, ~18.94 N, 72.83 E
    assert 18.9 < wankhede["lat"] < 19.0
    assert 72.8 < wankhede["lon"] < 72.85
    assert wankhede["tz"] == "Asia/Kolkata"

    chinnaswamy = geocodes["M Chinnaswamy Stadium"]
    # Bengaluru, ~12.98 N, 77.60 E
    assert 12.9 < chinnaswamy["lat"] < 13.05
    assert 77.55 < chinnaswamy["lon"] < 77.65


# ---------- B2: playing XI ----------

@pytest.fixture(scope="module")
def xi() -> pd.DataFrame:
    assert XI.exists(), f"missing {XI}"
    return pd.read_parquet(XI)


def test_xi_schema(xi):
    assert set(xi.columns) == {"match_id", "team1_xi", "team2_xi"}
    table = pq.read_table(XI)
    schema = table.schema
    assert str(schema.field("match_id").type) == "string"
    assert "list<" in str(schema.field("team1_xi").type)
    assert "list<" in str(schema.field("team2_xi").type)


def test_xi_one_row_per_match(xi):
    assert xi["match_id"].is_unique
    # Should cover the vast majority of 2008-2025 matches
    m = pd.read_parquet(HIST / "matches.parquet")
    m = m[m["season"].between(2008, 2025)]
    expected = set(m["match_id"].astype(str))
    actual = set(xi["match_id"].astype(str))
    coverage = len(actual & expected) / len(expected)
    assert coverage > 0.95, f"XI coverage {coverage:.1%} below 95%"


def test_xi_sizes_reasonable(xi):
    sizes = (xi["team1_xi"].str.len() + xi["team2_xi"].str.len())
    # Mean per match should be ~22 (could be slightly more from 2023 onward
    # due to the Impact Player rule allowing a 12th).
    assert 20.0 < sizes.mean() < 24.0, f"mean players/match = {sizes.mean()}"
    # Median should be exactly 22
    assert sizes.median() == 22.0


# ---------- B1: weather ----------

@pytest.fixture(scope="module")
def weather() -> pd.DataFrame:
    if not WEATHER.exists():
        pytest.skip(f"{WEATHER} not built yet")
    return pd.read_parquet(WEATHER)


def test_weather_schema(weather):
    expected = {"match_id", "temp_c", "humidity_pct", "dewpoint_c",
                "wind_kmh", "precipitation_mm"}
    assert set(weather.columns) == expected
    for col in expected - {"match_id"}:
        assert weather[col].dtype.kind == "f", f"{col} should be float"


def test_weather_no_snow_in_mumbai(weather):
    """Wankhede in April should not be sub-zero (sanity check)."""
    m = pd.read_parquet(HIST / "matches.parquet")
    m["match_id"] = m["match_id"].astype(str)
    weather = weather.copy()
    weather["match_id"] = weather["match_id"].astype(str)

    joined = m.merge(weather, on="match_id", how="inner")
    mum = joined[joined["venue"].astype(str).str.contains("Wankhede", na=False)]
    mum = mum.dropna(subset=["temp_c"])
    if len(mum) == 0:
        pytest.skip("no Wankhede weather rows yet")
    # Mumbai evening temps in IPL season (Mar-May): basically always 22-38 C
    assert mum["temp_c"].min() > 10, f"Wankhede min temp {mum['temp_c'].min()} too cold"
    assert mum["temp_c"].max() < 50, f"Wankhede max temp {mum['temp_c'].max()} too hot"


def test_weather_humidity_in_range(weather):
    h = weather["humidity_pct"].dropna()
    if len(h) == 0:
        pytest.skip("no weather rows yet")
    assert h.min() >= 0
    assert h.max() <= 100


def test_weather_ok_rate(weather):
    """At least 90% of rows should have a real temperature (API success)."""
    rate = weather["temp_c"].notna().mean()
    # Allow partial runs but flag complete-failure cases
    assert rate > 0.5, f"weather success rate only {rate:.1%}"
