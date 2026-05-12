"""v6 Phase B3 — venue geocodes seed.

Hand-curated lat/lon/city/country/tz for every distinct venue string seen in
matches.parquet (2008-2025). Keyed by the exact venue string (preserving the
",<city>" suffix variants used by cricsheet).

Coordinates sourced from Wikipedia / OpenStreetMap (free, no API). Timezones
are IANA names suitable for pandas/zoneinfo.

CLI: `python -m ml.src.v6.build_b3` writes
`ml/data/historical/v6/venue_geocodes.json` and prints a coverage summary.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import pandas as pd

from ml.src.v6._io import HIST, V6_DIR, load_matches

OUTPUT = V6_DIR / "venue_geocodes.json"


# Hand-curated; one entry per distinct venue string in matches.parquet.
# Where cricsheet uses both a bare name and a ",<city>" variant for the same
# physical ground, both keys are present and point to the same coordinates.
VENUE_GEOCODES: dict[str, dict] = {
    # --- Delhi ---
    "Arun Jaitley Stadium": {
        "name": "Arun Jaitley Stadium",
        "lat": 28.6378, "lon": 77.2432,
        "city": "Delhi", "country": "India", "tz": "Asia/Kolkata",
    },
    "Arun Jaitley Stadium, Delhi": {
        "name": "Arun Jaitley Stadium",
        "lat": 28.6378, "lon": 77.2432,
        "city": "Delhi", "country": "India", "tz": "Asia/Kolkata",
    },
    "Feroz Shah Kotla": {
        # Same ground as Arun Jaitley; pre-2019 name
        "name": "Feroz Shah Kotla",
        "lat": 28.6378, "lon": 77.2432,
        "city": "Delhi", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- Cuttack ---
    "Barabati Stadium": {
        "name": "Barabati Stadium",
        "lat": 20.4738, "lon": 85.8830,
        "city": "Cuttack", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- Guwahati ---
    "Barsapara Cricket Stadium, Guwahati": {
        "name": "Barsapara Cricket Stadium",
        "lat": 26.1158, "lon": 91.8061,
        "city": "Guwahati", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- Lucknow ---
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow": {
        "name": "Ekana Cricket Stadium",
        "lat": 26.7967, "lon": 81.0257,
        "city": "Lucknow", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- Brabourne, Mumbai ---
    "Brabourne Stadium": {
        "name": "Brabourne Stadium",
        "lat": 18.9356, "lon": 72.8243,
        "city": "Mumbai", "country": "India", "tz": "Asia/Kolkata",
    },
    "Brabourne Stadium, Mumbai": {
        "name": "Brabourne Stadium",
        "lat": 18.9356, "lon": 72.8243,
        "city": "Mumbai", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- South Africa 2009 venues ---
    "Buffalo Park": {
        "name": "Buffalo Park",
        "lat": -32.9986, "lon": 27.8650,
        "city": "East London", "country": "South Africa", "tz": "Africa/Johannesburg",
    },
    "De Beers Diamond Oval": {
        "name": "De Beers Diamond Oval",
        "lat": -28.7383, "lon": 24.7732,
        "city": "Kimberley", "country": "South Africa", "tz": "Africa/Johannesburg",
    },
    "Kingsmead": {
        "name": "Kingsmead",
        "lat": -29.8281, "lon": 31.0307,
        "city": "Durban", "country": "South Africa", "tz": "Africa/Johannesburg",
    },
    "New Wanderers Stadium": {
        "name": "New Wanderers Stadium",
        "lat": -26.1399, "lon": 28.0394,
        "city": "Johannesburg", "country": "South Africa", "tz": "Africa/Johannesburg",
    },
    "Newlands": {
        "name": "Newlands",
        "lat": -33.9719, "lon": 18.4628,
        "city": "Cape Town", "country": "South Africa", "tz": "Africa/Johannesburg",
    },
    "OUTsurance Oval": {
        "name": "OUTsurance Oval (Mangaung Oval)",
        "lat": -29.1167, "lon": 26.2117,
        "city": "Bloemfontein", "country": "South Africa", "tz": "Africa/Johannesburg",
    },
    "St George's Park": {
        "name": "St George's Park",
        "lat": -33.9456, "lon": 25.5994,
        "city": "Port Elizabeth", "country": "South Africa", "tz": "Africa/Johannesburg",
    },
    "SuperSport Park": {
        "name": "SuperSport Park",
        "lat": -25.7494, "lon": 28.2272,
        "city": "Centurion", "country": "South Africa", "tz": "Africa/Johannesburg",
    },

    # --- DY Patil, Mumbai ---
    "Dr DY Patil Sports Academy": {
        "name": "Dr. D.Y. Patil Sports Academy",
        "lat": 19.0436, "lon": 73.0277,
        "city": "Mumbai", "country": "India", "tz": "Asia/Kolkata",
    },
    "Dr DY Patil Sports Academy, Mumbai": {
        "name": "Dr. D.Y. Patil Sports Academy",
        "lat": 19.0436, "lon": 73.0277,
        "city": "Mumbai", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- Visakhapatnam ---
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": {
        "name": "ACA-VDCA Cricket Stadium",
        "lat": 17.7333, "lon": 83.3236,
        "city": "Visakhapatnam", "country": "India", "tz": "Asia/Kolkata",
    },
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": {
        "name": "ACA-VDCA Cricket Stadium",
        "lat": 17.7333, "lon": 83.3236,
        "city": "Visakhapatnam", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- UAE ---
    "Dubai International Cricket Stadium": {
        "name": "Dubai International Cricket Stadium",
        "lat": 25.0476, "lon": 55.2197,
        "city": "Dubai", "country": "United Arab Emirates", "tz": "Asia/Dubai",
    },
    "Sharjah Cricket Stadium": {
        "name": "Sharjah Cricket Stadium",
        "lat": 25.3294, "lon": 55.4239,
        "city": "Sharjah", "country": "United Arab Emirates", "tz": "Asia/Dubai",
    },
    "Sheikh Zayed Stadium": {
        "name": "Sheikh Zayed Stadium",
        "lat": 24.4036, "lon": 54.7236,
        "city": "Abu Dhabi", "country": "United Arab Emirates", "tz": "Asia/Dubai",
    },
    "Zayed Cricket Stadium, Abu Dhabi": {
        "name": "Sheikh Zayed Stadium",
        "lat": 24.4036, "lon": 54.7236,
        "city": "Abu Dhabi", "country": "United Arab Emirates", "tz": "Asia/Dubai",
    },

    # --- Kolkata ---
    "Eden Gardens": {
        "name": "Eden Gardens",
        "lat": 22.5648, "lon": 88.3433,
        "city": "Kolkata", "country": "India", "tz": "Asia/Kolkata",
    },
    "Eden Gardens, Kolkata": {
        "name": "Eden Gardens",
        "lat": 22.5648, "lon": 88.3433,
        "city": "Kolkata", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- Kanpur ---
    "Green Park": {
        "name": "Green Park",
        "lat": 26.4747, "lon": 80.3458,
        "city": "Kanpur", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- Dharamsala ---
    "Himachal Pradesh Cricket Association Stadium": {
        "name": "HPCA Stadium",
        "lat": 32.2186, "lon": 76.3232,
        "city": "Dharamsala", "country": "India", "tz": "Asia/Kolkata",
    },
    "Himachal Pradesh Cricket Association Stadium, Dharamsala": {
        "name": "HPCA Stadium",
        "lat": 32.2186, "lon": 76.3232,
        "city": "Dharamsala", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- Indore ---
    "Holkar Cricket Stadium": {
        "name": "Holkar Stadium",
        "lat": 22.7239, "lon": 75.8767,
        "city": "Indore", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- Ranchi ---
    "JSCA International Stadium Complex": {
        "name": "JSCA International Stadium",
        "lat": 23.4239, "lon": 85.3941,
        "city": "Ranchi", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- Bengaluru ---
    "M Chinnaswamy Stadium": {
        "name": "M. Chinnaswamy Stadium",
        "lat": 12.9789, "lon": 77.5996,
        "city": "Bengaluru", "country": "India", "tz": "Asia/Kolkata",
    },
    "M Chinnaswamy Stadium, Bengaluru": {
        "name": "M. Chinnaswamy Stadium",
        "lat": 12.9789, "lon": 77.5996,
        "city": "Bengaluru", "country": "India", "tz": "Asia/Kolkata",
    },
    "M.Chinnaswamy Stadium": {
        "name": "M. Chinnaswamy Stadium",
        "lat": 12.9789, "lon": 77.5996,
        "city": "Bengaluru", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- Chennai ---
    "MA Chidambaram Stadium": {
        "name": "MA Chidambaram Stadium",
        "lat": 13.0628, "lon": 80.2792,
        "city": "Chennai", "country": "India", "tz": "Asia/Kolkata",
    },
    "MA Chidambaram Stadium, Chepauk": {
        "name": "MA Chidambaram Stadium",
        "lat": 13.0628, "lon": 80.2792,
        "city": "Chennai", "country": "India", "tz": "Asia/Kolkata",
    },
    "MA Chidambaram Stadium, Chepauk, Chennai": {
        "name": "MA Chidambaram Stadium",
        "lat": 13.0628, "lon": 80.2792,
        "city": "Chennai", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- New Chandigarh (Mullanpur) ---
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur": {
        "name": "Maharaja Yadavindra Singh Stadium",
        "lat": 30.7710, "lon": 76.6650,
        "city": "Mullanpur", "country": "India", "tz": "Asia/Kolkata",
    },
    "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh": {
        "name": "Maharaja Yadavindra Singh Stadium",
        "lat": 30.7710, "lon": 76.6650,
        "city": "Mullanpur", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- Pune ---
    "Maharashtra Cricket Association Stadium": {
        "name": "MCA Stadium (Gahunje)",
        "lat": 18.6781, "lon": 73.7344,
        "city": "Pune", "country": "India", "tz": "Asia/Kolkata",
    },
    "Maharashtra Cricket Association Stadium, Pune": {
        "name": "MCA Stadium (Gahunje)",
        "lat": 18.6781, "lon": 73.7344,
        "city": "Pune", "country": "India", "tz": "Asia/Kolkata",
    },
    "Subrata Roy Sahara Stadium": {
        # Older name of the Gahunje ground used in cricsheet for 2012-2013 IPL Pune fixtures
        "name": "MCA Stadium (Gahunje)",
        "lat": 18.6781, "lon": 73.7344,
        "city": "Pune", "country": "India", "tz": "Asia/Kolkata",
    },
    "Nehru Stadium": {
        # IPL "Nehru Stadium" matches were played in Kochi (KTK home, 2011)
        "name": "Jawaharlal Nehru Stadium",
        "lat": 9.9978, "lon": 76.3047,
        "city": "Kochi", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- Ahmedabad ---
    "Narendra Modi Stadium, Ahmedabad": {
        "name": "Narendra Modi Stadium",
        "lat": 23.0918, "lon": 72.5970,
        "city": "Ahmedabad", "country": "India", "tz": "Asia/Kolkata",
    },
    "Sardar Patel Stadium, Motera": {
        # Predecessor / old name of the same Motera site
        "name": "Sardar Patel Stadium (Motera)",
        "lat": 23.0918, "lon": 72.5970,
        "city": "Ahmedabad", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- Mohali ---
    "Punjab Cricket Association IS Bindra Stadium": {
        "name": "PCA Stadium (Mohali)",
        "lat": 30.6913, "lon": 76.7355,
        "city": "Mohali", "country": "India", "tz": "Asia/Kolkata",
    },
    "Punjab Cricket Association IS Bindra Stadium, Mohali": {
        "name": "PCA Stadium (Mohali)",
        "lat": 30.6913, "lon": 76.7355,
        "city": "Mohali", "country": "India", "tz": "Asia/Kolkata",
    },
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh": {
        "name": "PCA Stadium (Mohali)",
        "lat": 30.6913, "lon": 76.7355,
        "city": "Mohali", "country": "India", "tz": "Asia/Kolkata",
    },
    "Punjab Cricket Association Stadium, Mohali": {
        "name": "PCA Stadium (Mohali)",
        "lat": 30.6913, "lon": 76.7355,
        "city": "Mohali", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- Hyderabad ---
    "Rajiv Gandhi International Stadium": {
        "name": "Rajiv Gandhi International Stadium",
        "lat": 17.4067, "lon": 78.5536,
        "city": "Hyderabad", "country": "India", "tz": "Asia/Kolkata",
    },
    "Rajiv Gandhi International Stadium, Uppal": {
        "name": "Rajiv Gandhi International Stadium",
        "lat": 17.4067, "lon": 78.5536,
        "city": "Hyderabad", "country": "India", "tz": "Asia/Kolkata",
    },
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad": {
        "name": "Rajiv Gandhi International Stadium",
        "lat": 17.4067, "lon": 78.5536,
        "city": "Hyderabad", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- Rajkot ---
    "Saurashtra Cricket Association Stadium": {
        "name": "SCA Stadium",
        "lat": 22.2839, "lon": 70.7831,
        "city": "Rajkot", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- Jaipur ---
    "Sawai Mansingh Stadium": {
        "name": "Sawai Mansingh Stadium",
        "lat": 26.8983, "lon": 75.8131,
        "city": "Jaipur", "country": "India", "tz": "Asia/Kolkata",
    },
    "Sawai Mansingh Stadium, Jaipur": {
        "name": "Sawai Mansingh Stadium",
        "lat": 26.8983, "lon": 75.8131,
        "city": "Jaipur", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- Raipur ---
    "Shaheed Veer Narayan Singh International Stadium": {
        "name": "Shaheed Veer Narayan Singh International Stadium",
        "lat": 21.1928, "lon": 81.7117,
        "city": "Raipur", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- Nagpur ---
    "Vidarbha Cricket Association Stadium, Jamtha": {
        "name": "VCA Stadium (Jamtha)",
        "lat": 21.0917, "lon": 79.0428,
        "city": "Nagpur", "country": "India", "tz": "Asia/Kolkata",
    },

    # --- Mumbai ---
    "Wankhede Stadium": {
        "name": "Wankhede Stadium",
        "lat": 18.9389, "lon": 72.8258,
        "city": "Mumbai", "country": "India", "tz": "Asia/Kolkata",
    },
    "Wankhede Stadium, Mumbai": {
        "name": "Wankhede Stadium",
        "lat": 18.9389, "lon": 72.8258,
        "city": "Mumbai", "country": "India", "tz": "Asia/Kolkata",
    },
}


def build() -> dict[str, dict]:
    matches = load_matches()
    venues = sorted(v for v in matches["venue"].dropna().unique())

    missing = [v for v in venues if v not in VENUE_GEOCODES]
    extra = [v for v in VENUE_GEOCODES if v not in set(venues)]

    if missing:
        raise RuntimeError(
            f"Missing geocodes for {len(missing)} venue(s):\n  "
            + "\n  ".join(repr(v) for v in missing)
        )

    out = {v: VENUE_GEOCODES[v] for v in venues}
    return out, missing, extra


def write_atomic(payload: dict, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def main() -> int:
    payload, missing, extra = build()
    write_atomic(payload, OUTPUT)
    print(f"wrote {OUTPUT}  ({len(payload)} venues)")
    if extra:
        print(f"  note: {len(extra)} curated entries not present in matches.parquet (left in for safety)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
