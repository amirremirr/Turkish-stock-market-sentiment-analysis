"""
Google Trends fetcher (zoom-out data source: public attention).

Pulls daily search interest in Turkey for economic-anxiety terms and stores it
in external_series. This measures what the PUBLIC is worried about — a different
lens from press sentiment (which is politically biased) or market prices.

Terms are pulled in ONE payload so Google's 0-100 index is comparable across
them. pytrends is unofficial and rate-limited; run occasionally, not in the
cloud loop.

Usage:  python fetch_gtrends.py
"""

import sys
import time

import pandas as pd

import database as db
from config import DB_PATH

TERMS = ["dolar", "enflasyon", "kriz", "zam", "faiz"]   # <=5 (Google's limit per payload)
START = "2026-03-01"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from pytrends.request import TrendReq
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    pt = TrendReq(hl="tr-TR", tz=180)
    for attempt in range(3):
        try:
            pt.build_payload(TERMS, timeframe=f"{START} {end}", geo="TR")
            df = pt.interest_over_time()
            break
        except Exception as exc:
            print(f"  pytrends attempt {attempt+1} failed ({type(exc).__name__}); retrying...")
            time.sleep(20)
    else:
        print("Google Trends unavailable (rate-limited). Try again later.")
        return

    if "isPartial" in df.columns:
        df = df[~df["isPartial"]] if df["isPartial"].dtype == bool else df.drop(columns=["isPartial"])
        df = df.drop(columns=[c for c in ["isPartial"] if c in df.columns])

    db.init_db(DB_PATH)
    rows = []
    for term in TERMS:
        for dt, val in df[term].items():
            rows.append((dt.strftime("%Y-%m-%d"), f"gt_{term}", float(val)))
    n = db.upsert_external_series(rows, db_path=DB_PATH)
    print(f"Stored {n} Google-Trends rows ({len(TERMS)} terms, {len(df)} days).")
    print("Latest search interest (0-100):")
    print(df.tail(3)[TERMS].to_string())


if __name__ == "__main__":
    main()
