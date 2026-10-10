"""Price providers, immutable bar snapshots, and the analysis panel.

Three rules run through this module:

* **A missing price is missing.** A ticker with no bar on a session has no
  return on that session and none on the next (which would otherwise span two
  days and be read as one). Nothing is filled, forward or with zero.
* **Multi-day returns use adjusted prices.** Close-to-close returns come from
  the dividend-and-split-adjusted series. Same-day quantities (open to close,
  the overnight gap) use the unadjusted pair and are withheld on any date with
  a recorded corporate action, where the two closes are not comparable.
* **An unexplained jump is not a return.** A one-session move larger than
  ``JUMP_THRESHOLD`` with no recorded action is more likely an unadjusted
  capital change than a price move the exchange's limits would allow. It is
  flagged and set to missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

from stock_research import store
from stock_research.calendar import SessionCalendar

BAR_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume", "dividend", "split_ratio"]
PRICE_RULE_VERSION = "stock-price-rules-v1"
#: Above any daily price limit Borsa Istanbul has applied to equities, so a
#: close-to-close move this large without a corporate action is a data defect.
JUMP_THRESHOLD = 0.25

ORIGIN_REAL = "real_historical"
ORIGIN_FIXTURE = "development_fixture"
ORIGIN_SYNTHETIC = "synthetic"


class PriceProvider(Protocol):
    name: str
    origin: str

    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Daily bars indexed by ISO date with ``BAR_COLUMNS``; empty if none."""


class YahooProvider:
    """Yahoo Finance daily bars. ``Close`` is split-adjusted, ``Adj Close``
    also dividend-adjusted. Delisted tickers are generally not served."""

    name = "yahoo-finance"
    origin = ORIGIN_REAL

    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        import yfinance as yf

        raw = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False, actions=True)
        if raw is None or raw.empty:
            return pd.DataFrame(columns=BAR_COLUMNS)
        frame = pd.DataFrame({
            "open": raw["Open"], "high": raw["High"], "low": raw["Low"],
            "close": raw["Close"], "adj_close": raw["Adj Close"],
            "volume": raw["Volume"],
            "dividend": raw.get("Dividends", 0.0),
            "split_ratio": raw.get("Stock Splits", 0.0),
        })
        frame.index = [stamp.strftime("%Y-%m-%d") for stamp in raw.index]
        return frame[~frame.index.duplicated(keep="last")]


class FrameProvider:
    """Serves frames held in memory: fixtures and synthetic panels."""

    def __init__(self, frames: Dict[str, pd.DataFrame], *, origin: str = ORIGIN_SYNTHETIC,
                 name: str = "in-memory"):
        self.frames, self.origin, self.name = frames, origin, name

    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        frame = self.frames.get(ticker)
        if frame is None:
            return pd.DataFrame(columns=BAR_COLUMNS)
        return frame.loc[(frame.index >= start) & (frame.index < end)]


@dataclass(frozen=True)
class Availability:
    """Whether a dataset can be used, and why not when it cannot."""

    dataset: str
    available: bool
    reason: str


def market_cap_availability() -> Availability:
    return Availability(
        "point_in_time_market_cap", False,
        "No verifiable source of historical shares outstanding or free float "
        "for BIST is configured. Today's share count times a historical price "
        "would be look-ahead, so size-dependent tests are blocked.",
    )


def intraday_availability() -> Availability:
    return Availability(
        "intraday_prices", False,
        "Only daily bars are available; intraday lead-lag cannot be measured.",
    )


# -- Snapshots ---------------------------------------------------------------
def ingest(tickers: Iterable[str], start: str, end: str, provider: PriceProvider,
           snapshot_id: str, db_path=None) -> Dict[str, int]:
    """Fetch and store bars under *snapshot_id*. Already-stored tickers are skipped."""

    store.init_db(db_path)
    counts = {"ok": 0, "no_data": 0, "error": 0, "skipped": 0}
    for ticker in tickers:
        with store.connect(db_path) as con:
            if con.execute(
                "SELECT 1 FROM sr_price_availability WHERE snapshot_id=? AND ticker=?",
                (snapshot_id, ticker),
            ).fetchone():
                counts["skipped"] += 1
                continue
        status, detail, frame = "ok", None, pd.DataFrame(columns=BAR_COLUMNS)
        try:
            frame = provider.fetch(ticker, start, end)
            if frame.empty:
                status = "no_data"
        except Exception as exc:                      # provider failures are data
            status, detail = "error", f"{type(exc).__name__}: {exc}"[:300]
        now = store.now_iso()
        with store.connect(db_path) as con:
            for day, row in frame.iterrows():
                con.execute(
                    "INSERT OR IGNORE INTO sr_raw_price_bars VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (snapshot_id, ticker, day, *[_num(row[c]) for c in BAR_COLUMNS],
                     provider.name, now),
                )
            con.execute(
                "INSERT OR REPLACE INTO sr_price_availability VALUES (?,?,?,?,?,?,?,?,?)",
                (snapshot_id, ticker, status, len(frame),
                 frame.index.min() if len(frame) else None,
                 frame.index.max() if len(frame) else None, detail, provider.name, now),
            )
        counts[status] += 1
    return counts


def _num(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


# -- Validation --------------------------------------------------------------
def validate_bars(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """Structural defects in one ticker's bars. Each is a row, never a raise."""

    issues: List[Dict[str, Any]] = []
    if frame.empty:
        return issues

    def flag(mask: pd.Series, code: str) -> None:
        for day in frame.index[mask.fillna(False).to_numpy()]:
            issues.append({"date": day, "issue": code})

    prices = frame[["open", "high", "low", "close"]]
    flag((prices <= 0).any(axis=1), "non_positive_price")
    flag(frame["high"] < frame["low"], "high_below_low")
    tolerance = 1e-6 * frame["close"].abs()
    flag((frame["close"] > frame["high"] + tolerance) | (frame["close"] < frame["low"] - tolerance),
         "close_outside_range")
    flag(frame["open"].isna() & frame["close"].notna(), "missing_open")
    flag(frame["volume"].fillna(0) <= 0, "zero_or_missing_volume")
    if not frame.index.is_monotonic_increasing:
        issues.append({"date": None, "issue": "dates_not_sorted"})
    return issues


# -- Panel -------------------------------------------------------------------
@dataclass
class Panel:
    """Bars aligned to the benchmark's sessions, with derived return columns.

    Every frame is indexed by the full session list; a session the ticker did
    not trade is a row of NaN, so a missing day is visible instead of absent.
    """

    calendar: SessionCalendar
    bars: Dict[str, pd.DataFrame]
    benchmark: str
    origin: str
    snapshot_id: str
    issues: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def frame(self, ticker: str) -> Optional[pd.DataFrame]:
        return self.bars.get(ticker)

    @property
    def market(self) -> pd.DataFrame:
        return self.bars[self.benchmark]


def derive(frame: pd.DataFrame, sessions: Sequence[str]) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Align one ticker to *sessions* and add returns, with missing states."""

    issues = validate_bars(frame)
    aligned = frame.reindex(list(sessions)).astype(float)
    action = (aligned["dividend"].fillna(0) != 0) | (aligned["split_ratio"].fillna(0) != 0)

    previous = aligned["adj_close"].shift(1)
    ret = aligned["adj_close"] / previous - 1.0          # NaN if either side is missing
    jump = ret.abs() > JUMP_THRESHOLD
    unexplained = jump & ~action
    for day in aligned.index[unexplained.fillna(False).to_numpy()]:
        issues.append({"date": day, "issue": "unexplained_jump"})
    ret = ret.mask(unexplained)

    bad = {i["date"] for i in issues if i["issue"] in (
        "non_positive_price", "high_below_low", "close_outside_range")}
    # A return that starts or ends on a structurally bad bar is not a return.
    bad_mask = pd.Series(aligned.index.isin(bad), index=aligned.index)
    ret = ret.mask(bad_mask | bad_mask.shift(1, fill_value=False))

    aligned["ret"] = ret
    aligned["corporate_action"] = action
    prior_close = aligned["close"].shift(1)
    same_day_ok = ~action & ~aligned.index.isin(bad)
    aligned["gap"] = np.log(aligned["open"] / prior_close).where(same_day_ok)
    aligned["intraday"] = np.log(aligned["close"] / aligned["open"]).where(same_day_ok)
    aligned["raw_ret"] = (aligned["close"] / prior_close - 1.0).where(same_day_ok)
    aligned["turnover"] = aligned["close"] * aligned["volume"]
    return aligned, issues


def build_panel(frames: Dict[str, pd.DataFrame], benchmark: str, *, origin: str,
                snapshot_id: str) -> Panel:
    if benchmark not in frames or frames[benchmark].empty:
        raise ValueError(f"benchmark {benchmark!r} has no bars; no calendar can be built")
    market = frames[benchmark]
    calendar = SessionCalendar(market.index[market["close"].notna()])
    bars, issues = {}, {}
    for ticker, frame in frames.items():
        if frame is None or frame.empty:
            continue
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        bars[ticker], found = derive(frame, calendar.sessions)
        if found:
            issues[ticker] = found
    return Panel(calendar, bars, benchmark, origin, snapshot_id, issues)


def load_panel(snapshot_id: str, benchmark: str, db_path=None, *,
               origin: str = ORIGIN_REAL) -> Panel:
    with store.connect(db_path) as con:
        table = pd.read_sql_query(
            "SELECT ticker, date, open, high, low, close, adj_close, volume, "
            "dividend, split_ratio FROM sr_raw_price_bars WHERE snapshot_id = ? "
            "ORDER BY ticker, date", con, params=(snapshot_id,),
        )
    frames = {
        ticker: group.drop(columns="ticker").set_index("date")
        for ticker, group in table.groupby("ticker")
    }
    return build_panel(frames, benchmark, origin=origin, snapshot_id=snapshot_id)
