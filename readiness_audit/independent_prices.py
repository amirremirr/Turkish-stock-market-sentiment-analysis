"""Second source for prices and capital actions: Is Yatirim's public data.

    python -m readiness_audit.independent_prices fetch      # network, cached
    python -m readiness_audit.independent_prices compare

For every stock in the price panel this fetches Is Yatirim's daily closes
(raw and adjusted) and its list of capital increases and dividends, and
compares them with the Yahoo bars the study uses: which corporate actions each
source records, and whether the two adjusted one-session returns agree.

Only single-session returns are compared, for all stocks and all sessions
alike. No disclosure of the sample is looked up and no event window is built.
The fetched series stay in the audit output directory, which is not committed.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from readiness_audit import price_checks
from readiness_audit import snapshot as snap
from stock_research import pipeline
from stock_research.config import BENCHMARK_TICKER
from stock_research.data import prices

CACHE = snap.AUDIT_OUT / "isyatirim"
BASE = "https://www.isyatirim.com.tr/_layouts/15/IsYatirim.Website"
HEADERS = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest",
           "Content-Type": "application/json; charset=UTF-8"}
ACTION_TYPES = {"01": "rights_issue", "02": "bonus_issue", "03": "rights_and_bonus",
                "04": "cash_dividend"}


def _code(ticker: str) -> str:
    return ticker.split(".")[0]


def fetch(pause: float = 0.4) -> Dict[str, int]:
    import requests

    CACHE.mkdir(parents=True, exist_ok=True)
    bars = price_checks.raw_bars()
    tickers = sorted(t for t in bars["ticker"].unique() if t != BENCHMARK_TICKER)
    start = pd.Timestamp(pipeline.PRICE_START).strftime("%d-%m-%Y")
    end = pd.Timestamp(pipeline.PRICE_END).strftime("%d-%m-%Y")
    counts = {"ok": 0, "cached": 0, "error": 0}
    for ticker in tickers:
        target = CACHE / f"{_code(ticker)}.json"
        if target.exists():
            counts["cached"] += 1
            continue
        try:
            series = requests.get(
                f"{BASE}/Common/Data.aspx/HisseTekil", headers={"User-Agent": "Mozilla/5.0"},
                params={"hisse": _code(ticker), "startdate": start, "enddate": end}, timeout=60)
            time.sleep(pause)
            actions = requests.post(
                f"{BASE}/StockInfo/CompanyInfoAjax.aspx/GetSermayeArttirimlari", headers=HEADERS,
                data=json.dumps({"hisseKodu": _code(ticker), "hisseTanimKodu": "", "yil": 0,
                                 "zaman": "HEPSI", "endeksKodu": "09", "sektorKodu": ""}),
                timeout=60)
            time.sleep(pause)
            payload = {
                "prices": [{"date": r["HGDG_TARIH"], "adjusted": r["HGDG_KAPANIS"],
                            "raw": r.get("HG_KAPANIS")} for r in (series.json().get("value") or [])],
                "actions": [{"type": r.get("SHHE_TIP_KODU"), "date_ms": r.get("SHHE_TARIH"),
                             "rights_pct": r.get("SHHE_BDLI_ORAN"),
                             "bonus_internal_pct": r.get("SHHE_BDSZ_IK_ORAN"),
                             "bonus_dividend_pct": r.get("SHHE_BDSZ_TM_ORAN"),
                             "cash_dividend_pct": r.get("SHHE_NAKIT_TM_ORAN"),
                             "adjustment_ratio": r.get("SHHE_FIYAT_AYAR_OR")}
                            for r in json.loads(actions.json().get("d") or "[]")],
            }
            target.write_text(json.dumps(payload), encoding="utf-8")
            counts["ok"] += 1
        except Exception as exc:                         # recorded, not fatal
            counts["error"] += 1
            print(ticker, type(exc).__name__, str(exc)[:120], flush=True)
    return counts


def _load(ticker: str):
    path = CACHE / f"{_code(ticker)}.json"
    if not path.exists():
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    series = pd.DataFrame(payload["prices"])
    if len(series):
        series["date"] = pd.to_datetime(series["date"], format="%d-%m-%Y").dt.strftime("%Y-%m-%d")
        series = series.drop_duplicates("date").set_index("date").sort_index()
    actions = pd.DataFrame(payload["actions"])
    if len(actions):
        # Stamps are local midnight (UTC+3) in epoch milliseconds.
        actions = actions[actions["date_ms"].fillna(-1) > 0].copy()     # undated rows
        actions["date"] = (pd.to_datetime(actions["date_ms"], unit="ms")
                           + pd.Timedelta(hours=3)).dt.strftime("%Y-%m-%d")
        actions["kind"] = actions["type"].map(ACTION_TYPES).fillna(actions["type"])
        actions = actions[(actions["date"] >= pipeline.PRICE_START)
                          & (actions["date"] < pipeline.PRICE_END)]
    return series, actions


def compare() -> Dict[str, Any]:
    panel = prices.load_panel(f"yahoo-{pipeline.PRICE_START}-{pipeline.PRICE_END}",
                              BENCHMARK_TICKER, snap.SNAPSHOT_DB)
    rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []
    covered = no_second_source = 0
    for ticker, frame in panel.bars.items():
        if ticker == BENCHMARK_TICKER:
            continue
        series, actions = _load(ticker)
        if series is None or not len(series):
            no_second_source += 1
            continue
        covered += 1
        other = series.reindex(frame.index)
        second = other["adjusted"] / other["adjusted"].shift(1) - 1
        second_raw = other["raw"] / other["raw"].shift(1) - 1
        table = pd.DataFrame({
            "ticker": ticker, "date": frame.index, "ret": frame["ret"].to_numpy(),
            "second": second.to_numpy(), "second_raw": second_raw.to_numpy(),
            "yahoo_action": frame["corporate_action"].to_numpy(),
            "yahoo_split": (frame["split_ratio"].fillna(0) != 0).to_numpy(),
            "yahoo_dividend": (frame["dividend"].fillna(0) != 0).to_numpy(),
        })
        theirs = actions if actions is not None and len(actions) else pd.DataFrame(
            columns=["date", "kind", "adjustment_ratio"])
        table["second_action"] = table["date"].isin(set(theirs["date"]))
        rows.append(table)
        for _, action in theirs.iterrows():
            position = panel.calendar.index(action["date"])
            near = False
            if position is not None:
                window = frame.iloc[max(0, position - 1):position + 2]
                near = bool(window["corporate_action"].any())
            action_rows.append({"ticker": ticker, "date": action["date"], "kind": action["kind"],
                                "adjustment_ratio": action["adjustment_ratio"],
                                "session": position is not None,
                                "yahoo_records_action_within_1_session": near})
    data = pd.concat(rows, ignore_index=True)
    both = data.dropna(subset=["ret", "second"]).copy()
    both["diff"] = both["ret"] - both["second"]
    actions = pd.DataFrame(action_rows)

    def share(mask) -> float:
        return float(mask.mean()) if len(mask) else float("nan")

    plain = both[~both["yahoo_action"] & ~both["second_action"]]
    split_days = both[both["yahoo_split"]]
    dividend_days = both[both["yahoo_dividend"] & ~both["yahoo_split"]]
    capital = actions[actions["kind"].isin(["rights_issue", "bonus_issue", "rights_and_bonus"])]
    dividends = actions[actions["kind"] == "cash_dividend"]
    wrong = both[both["diff"].abs() > 0.01]
    out = {
        "stocks_in_panel": int(len(panel.bars) - 1), "stocks_with_second_source": covered,
        "stocks_without_second_source": no_second_source,
        "returns_compared": int(len(both)),
        "ordinary_days": {
            "compared": int(len(plain)),
            "median_abs_difference": float(plain["diff"].abs().median()),
            "share_differing_over_0.2pct": share(plain["diff"].abs() > 0.002),
            "share_differing_over_1pct": share(plain["diff"].abs() > 0.01),
        },
        "yahoo_split_days": {
            "compared": int(len(split_days)),
            "differing_over_1pct": int((split_days["diff"].abs() > 0.01).sum()),
            "differing_over_5pct": int((split_days["diff"].abs() > 0.05).sum()),
        },
        "yahoo_dividend_days": {
            "compared": int(len(dividend_days)),
            "differing_over_1pct": int((dividend_days["diff"].abs() > 0.01).sum()),
        },
        "second_source_capital_actions": {
            "total": int(len(capital)), "by_kind": capital["kind"].value_counts().to_dict(),
            "recorded_by_yahoo": int(capital["yahoo_records_action_within_1_session"].sum()),
            "not_recorded_by_yahoo": int((~capital["yahoo_records_action_within_1_session"]).sum()),
        },
        "second_source_cash_dividends": {
            "total": int(len(dividends)),
            "recorded_by_yahoo": int(dividends["yahoo_records_action_within_1_session"].sum()),
            "not_recorded_by_yahoo": int((~dividends["yahoo_records_action_within_1_session"]).sum()),
        },
        "returns_kept_by_cleaning_that_differ_over_1pct": int(len(wrong)),
        "of_which_on_a_recorded_action_day": int((wrong["yahoo_action"] | wrong["second_action"]).sum()),
        "returns_kept_that_differ_over_5pct": int((both["diff"].abs() > 0.05).sum()),
    }
    both.to_csv(snap.AUDIT_OUT / "return_comparison.csv", index=False)
    actions.to_csv(snap.AUDIT_OUT / "second_source_actions.csv", index=False)
    wrong.sort_values("diff", key=np.abs, ascending=False).to_csv(
        snap.AUDIT_OUT / "return_disagreements.csv", index=False)
    (snap.AUDIT_OUT / "independent_prices.json").write_text(json.dumps(out, indent=1),
                                                            encoding="utf-8")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    what = sys.argv[1] if len(sys.argv) > 1 else "compare"
    print(json.dumps(fetch() if what == "fetch" else compare(), indent=1))
