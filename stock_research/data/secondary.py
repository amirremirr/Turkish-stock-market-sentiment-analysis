"""A second price source, kept to check the first.

The readiness audit compared the study's provider with a second one over
104,535 daily returns and found the first wrong in two ways the cleaning rules
did not catch: on sixteen dates its "close" was not the closing price for a
large share of stocks, and four capital increases carried a wrong adjustment
factor. Protocol v3 therefore stores a second provider's closes and its list
of capital actions, and ``data.prices`` admits a return only where the two
agree.

The second source is evidence about the first, never a substitute for it: no
return is taken from it.

Provider. Is Yatirim's public data service gives, for a stock, its daily
closes raw and adjusted and its capital increases and dividends with dates and
adjustment ratios, and for an index its closing values. It is a broker's
service, not the exchange's.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, List, Optional, Protocol

import numpy as np
import pandas as pd

from stock_research import store

SOURCE = "isyatirim"
BAR_COLUMNS = ["close_raw", "close_adjusted"]
ACTION_COLUMNS = ["action_date", "kind", "rights_pct", "bonus_pct", "cash_dividend_pct",
                  "adjustment_ratio"]
KIND_RIGHTS, KIND_BONUS, KIND_BOTH, KIND_DIVIDEND, KIND_OTHER = (
    "rights_issue", "bonus_issue", "rights_and_bonus", "cash_dividend", "other")
CAPITAL_KINDS = (KIND_RIGHTS, KIND_BONUS, KIND_BOTH)
_KINDS = {"01": KIND_RIGHTS, "02": KIND_BONUS, "03": KIND_BOTH, "04": KIND_DIVIDEND}


class SecondaryProvider(Protocol):
    name: str

    def fetch(self, ticker: str, start: str, end: str, *, is_index: bool = False) -> Dict[str, Any]:
        """``{"bars": frame indexed by ISO date with BAR_COLUMNS,
        "actions": list of dicts with ACTION_COLUMNS}``; empty where none."""


def _empty() -> Dict[str, Any]:
    return {"bars": pd.DataFrame(columns=BAR_COLUMNS), "actions": []}


class IsYatirimProvider:
    """Is Yatirim's public endpoints. A stock code is the ticker without its
    exchange suffix; the benchmark is asked for as an index."""

    name = SOURCE
    BASE = "https://www.isyatirim.com.tr/_layouts/15/IsYatirim.Website"

    def __init__(self, pause: float = 0.4):
        self.pause = pause

    @staticmethod
    def code(ticker: str) -> str:
        return ticker.split(".")[0]

    def fetch(self, ticker: str, start: str, end: str, *, is_index: bool = False) -> Dict[str, Any]:
        import requests

        agent = {"User-Agent": "Mozilla/5.0"}
        if is_index:
            stamp = lambda day, clock: pd.Timestamp(day).strftime("%Y%m%d") + clock   # noqa: E731
            response = requests.get(
                f"{self.BASE}/Common/ChartData.aspx/IndexHistoricalAll", headers=agent, timeout=60,
                params={"period": "1440", "from": stamp(start, "000000"),
                        "to": stamp(end, "235959"), "endeks": self.code(ticker)})
            time.sleep(self.pause)
            rows = (response.json() or {}).get("data") or []
            if not rows:
                return _empty()
            # Stamps are local midnight (UTC+3) in epoch milliseconds.
            days = [(pd.to_datetime(ms, unit="ms") + pd.Timedelta(hours=3)).strftime("%Y-%m-%d")
                    for ms, _ in rows]
            values = [float(v) for _, v in rows]
            frame = pd.DataFrame({"close_raw": values, "close_adjusted": values}, index=days)
            return {"bars": frame[~frame.index.duplicated(keep="last")], "actions": []}

        day = lambda value: pd.Timestamp(value).strftime("%d-%m-%Y")                   # noqa: E731
        series = requests.get(
            f"{self.BASE}/Common/Data.aspx/HisseTekil", headers=agent, timeout=60,
            params={"hisse": self.code(ticker), "startdate": day(start), "enddate": day(end)})
        time.sleep(self.pause)
        listed = requests.post(
            f"{self.BASE}/StockInfo/CompanyInfoAjax.aspx/GetSermayeArttirimlari", timeout=60,
            headers={**agent, "X-Requested-With": "XMLHttpRequest",
                     "Content-Type": "application/json; charset=UTF-8"},
            data=json.dumps({"hisseKodu": self.code(ticker), "hisseTanimKodu": "", "yil": 0,
                             "zaman": "HEPSI", "endeksKodu": "09", "sektorKodu": ""}))
        time.sleep(self.pause)
        values = (series.json() or {}).get("value") or []
        frame = pd.DataFrame({
            "close_raw": [row.get("HG_KAPANIS") for row in values],
            "close_adjusted": [row.get("HGDG_KAPANIS") for row in values],
        }, index=[pd.to_datetime(row["HGDG_TARIH"], format="%d-%m-%Y").strftime("%Y-%m-%d")
                  for row in values])
        actions = []
        for row in json.loads((listed.json() or {}).get("d") or "[]"):
            stamp_ms = row.get("SHHE_TARIH")
            if not stamp_ms or stamp_ms <= 0:
                continue
            when = (pd.to_datetime(stamp_ms, unit="ms") + pd.Timedelta(hours=3)).strftime("%Y-%m-%d")
            if not (start <= when < end):
                continue
            actions.append({
                "action_date": when, "kind": _KINDS.get(row.get("SHHE_TIP_KODU"), KIND_OTHER),
                "rights_pct": row.get("SHHE_BDLI_ORAN"),
                "bonus_pct": (row.get("SHHE_BDSZ_IK_ORAN") or 0) + (row.get("SHHE_BDSZ_TM_ORAN") or 0),
                "cash_dividend_pct": row.get("SHHE_NAKIT_TM_ORAN"),
                "adjustment_ratio": row.get("SHHE_FIYAT_AYAR_OR")})
        return {"bars": frame[~frame.index.duplicated(keep="last")], "actions": actions}


class FrameSecondary:
    """Serves frames held in memory: fixtures and synthetic panels."""

    def __init__(self, bars: Dict[str, pd.DataFrame],
                 actions: Optional[Dict[str, List[Dict[str, Any]]]] = None, name: str = "in-memory"):
        self.bars, self.actions, self.name = bars, actions or {}, name

    def fetch(self, ticker: str, start: str, end: str, *, is_index: bool = False) -> Dict[str, Any]:
        frame = self.bars.get(ticker)
        if frame is None:
            return _empty()
        return {"bars": frame.loc[(frame.index >= start) & (frame.index < end), BAR_COLUMNS],
                "actions": list(self.actions.get(ticker, []))}


def from_primary(frames: Dict[str, pd.DataFrame], *, name: str = "in-memory") -> FrameSecondary:
    """A second source that agrees with primary bars exactly. For fixtures."""

    return FrameSecondary({
        ticker: pd.DataFrame({"close_raw": frame["close"], "close_adjusted": frame["adj_close"]})
        for ticker, frame in frames.items()}, name=name)


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def ingest(tickers: Iterable[str], start: str, end: str, provider: SecondaryProvider,
           snapshot_id: str, db_path=None, *, benchmark: Optional[str] = None) -> Dict[str, int]:
    """Fetch and store second-source closes and capital actions under
    *snapshot_id*. A ticker already asked for is skipped; one the provider
    cannot serve is recorded as such."""

    store.init_db(db_path)
    counts = {"ok": 0, "no_data": 0, "error": 0, "skipped": 0}
    for ticker in tickers:
        with store.connect(db_path) as con:
            if con.execute("SELECT 1 FROM sr_price_availability WHERE snapshot_id=? AND ticker=?",
                           (snapshot_id, ticker)).fetchone():
                counts["skipped"] += 1
                continue
        status, detail, found = "ok", None, _empty()
        try:
            found = provider.fetch(ticker, start, end, is_index=(ticker == benchmark))
            if found["bars"].empty:
                status = "no_data"
        except Exception as exc:                      # provider failures are data
            status, detail = "error", f"{type(exc).__name__}: {exc}"[:300]
        now = store.now_iso()
        frame = found["bars"]
        with store.connect(db_path) as con:
            for day, row in frame.iterrows():
                con.execute(
                    "INSERT OR IGNORE INTO sr_raw_price_bars_secondary VALUES (?,?,?,?,?,?,?)",
                    (snapshot_id, ticker, day, _number(row["close_raw"]),
                     _number(row["close_adjusted"]), provider.name, now))
            for action in found["actions"]:
                con.execute(
                    "INSERT OR IGNORE INTO sr_raw_capital_actions VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (snapshot_id, ticker, action["action_date"], action["kind"],
                     _number(action.get("rights_pct")), _number(action.get("bonus_pct")),
                     _number(action.get("cash_dividend_pct")),
                     _number(action.get("adjustment_ratio")), provider.name, now))
            con.execute(
                "INSERT OR REPLACE INTO sr_price_availability VALUES (?,?,?,?,?,?,?,?,?)",
                (snapshot_id, ticker, status, len(frame),
                 frame.index.min() if len(frame) else None,
                 frame.index.max() if len(frame) else None, detail, provider.name, now))
        counts[status] += 1
    return counts


def load(snapshot_id: str, db_path=None):
    """(closes by ticker, capital actions) of one second-source snapshot."""

    with store.connect(db_path) as con:
        bars = pd.read_sql_query(
            "SELECT ticker, date, close_raw, close_adjusted FROM sr_raw_price_bars_secondary "
            "WHERE snapshot_id = ? ORDER BY ticker, date", con, params=(snapshot_id,))
        actions = pd.read_sql_query(
            "SELECT ticker, action_date, kind, rights_pct, bonus_pct, cash_dividend_pct, "
            "adjustment_ratio FROM sr_raw_capital_actions WHERE snapshot_id = ? "
            "ORDER BY ticker, action_date, kind", con, params=(snapshot_id,))
    frames = {ticker: group.drop(columns="ticker").set_index("date")
              for ticker, group in bars.groupby("ticker")}
    return frames, actions
