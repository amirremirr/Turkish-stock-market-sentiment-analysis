"""
GDELT fetcher (zoom-out data source: the WORLD's media view of Turkey).

Pulls a daily timeline of average media TONE and article VOLUME for Turkey's
economy from GDELT's global news database (the whole world's press, not just
Turkish outlets) and stores it in external_series. Free, no key.

This is the "foreign investors read global media" lens — how Turkey is covered
abroad, which may diverge from (and price differently than) the domestic press.

Usage:  python fetch_gdelt.py
"""

import sys
import time

import pandas as pd
import requests

import database as db
from config import DB_PATH

QUERY = '(Turkey OR Turkish) (economy OR lira OR inflation OR central bank)'
START = "20260301000000"
API = "https://api.gdeltproject.org/api/v2/doc/doc"


def _timeline(mode):
    for attempt in range(4):
        r = requests.get(API, params={
            "query": QUERY, "mode": mode, "format": "json",
            "startdatetime": START,
            "enddatetime": pd.Timestamp.today().strftime("%Y%m%d000000"),
            "timelinesmooth": "0",
        }, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 429:
            wait = 15 * (attempt + 1)
            print(f"    GDELT 429, waiting {wait}s ...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        break
    tl = r.json().get("timeline", [])
    if not tl:
        return {}
    out = {}
    for pt in tl[0]["data"]:
        d = pt["date"][:8]
        out[f"{d[:4]}-{d[4:6]}-{d[6:8]}"] = float(pt["value"])
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        tone = _timeline("TimelineTone")
        time.sleep(5)   # space the two calls to avoid rate-limiting
        vol = _timeline("TimelineVol")
    except Exception as exc:
        print(f"GDELT fetch failed: {type(exc).__name__}: {exc}")
        return
    if not tone:
        print("GDELT returned no data.")
        return
    db.init_db(DB_PATH)
    rows = [(d, "gdelt_tone", v) for d, v in tone.items()]
    rows += [(d, "gdelt_volume", v) for d, v in vol.items()]
    n = db.upsert_external_series(rows, db_path=DB_PATH)
    print(f"Stored {n} GDELT rows (tone + volume, {len(tone)} days).")
    s = pd.Series(tone).sort_index()
    print(f"Global media tone on Turkey: mean {s.mean():+.2f} "
          f"(negative = negative coverage), latest {s.iloc[-1]:+.2f}")


if __name__ == "__main__":
    main()
