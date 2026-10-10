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
* **A move the exchange forbids is not a return.** Where a verified price
  limit applies, a one-session move beyond it cannot have happened; it means
  one of the two bars is wrong (a partial bar, an unrecorded capital change).
  It is flagged ``move_beyond_price_limit`` and set to missing.
* **A bar with no volume is not a trade.** Providers carry the last price
  forward on days a stock did not trade. Read as a bar, that is a fabricated
  zero return, so for stocks it is treated as a missing session.
* **A cancelled session never happened.** ``CANCELLED_SESSIONS`` lists dates
  whose trades the exchange annulled; their leftover bars are removed from
  every series before anything is computed.
* **A session the benchmark lacks is merged, and said to be.** The calendar is
  the benchmark's. If most stocks traded on a date the benchmark has no bar
  for, that date folds into the next session for stock and market alike: the
  close-to-close return there is a two-day return on both sides, which keeps
  abnormal returns valid, while same-day quantities (the gap, the
  prior-close move used for price limits) are withheld on the merged session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

from stock_research import store
from stock_research.calendar import SessionCalendar

BAR_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume", "dividend", "split_ratio"]
PRICE_RULE_VERSION = "stock-price-rules-v2"

#: Sessions whose trades were annulled by the exchange. A provider may still
#: serve a bar for them; using it would measure the next return from a price
#: that officially does not exist.
CANCELLED_SESSIONS: Dict[str, str] = {
    "2023-02-08": (
        "Borsa Istanbul halted trading after the 6 February earthquakes and "
        "cancelled all trades executed on 8 February 2023; the market reopened "
        "on 15 February. Source: Anadolu Agency, 'Turkish stock exchange reopens "
        "on Wednesday' (aa.com.tr/en/economy/turkish-stock-exchange-reopens-on-"
        "wednesday/2820232); AGBI/Reuters, 'Turkish bourse shuts for five days "
        "and cancels trades after quake'."
    ),
}
#: Share of stocks that must have traded on a date for it to count as a
#: session the benchmark is missing.
CALENDAR_GAP_SHARE = 0.5
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
    #: Dates most stocks traded but the benchmark has no bar for.
    calendar_gaps: List[str] = field(default_factory=list)
    #: Sessions that absorbed one or more of those dates.
    merged_sessions: List[str] = field(default_factory=list)

    def frame(self, ticker: str) -> Optional[pd.DataFrame]:
        return self.bars.get(ticker)

    @property
    def market(self) -> pd.DataFrame:
        return self.bars[self.benchmark]


def derive(frame: pd.DataFrame, sessions: Sequence[str], *, is_benchmark: bool = False,
           merged: Sequence[str] = ()) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Align one ticker to *sessions* and add returns, with missing states."""

    issues = validate_bars(frame)
    if not is_benchmark:
        # An index has no volume of its own to speak of; a stock without
        # volume did not trade, and its bar is the provider's carry-forward.
        idle = frame["volume"].isna() | (frame["volume"] <= 0)
        if idle.any():
            frame = frame.copy()
            frame.loc[idle, ["open", "high", "low", "close", "adj_close"]] = np.nan
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

    prior_close = aligned["close"].shift(1)
    same_day_ok = ~action & ~aligned.index.isin(bad) & ~aligned.index.isin(list(merged))
    raw_ret = (aligned["close"] / prior_close - 1.0).where(same_day_ok)

    if not is_benchmark:
        from stock_research.limits import TOLERANCE, limit_on

        limit = pd.Series([limit_on(day) for day in aligned.index], index=aligned.index,
                          dtype=float)
        forbidden = (raw_ret.abs() > limit + TOLERANCE).fillna(False)
        for day in aligned.index[forbidden.to_numpy()]:
            issues.append({"date": day, "issue": "move_beyond_price_limit"})
        ret = ret.mask(forbidden)
        same_day_ok = same_day_ok & ~forbidden
        raw_ret = raw_ret.where(same_day_ok)

    aligned["ret"] = ret
    aligned["corporate_action"] = action
    aligned["gap"] = np.log(aligned["open"] / prior_close).where(same_day_ok)
    aligned["intraday"] = np.log(aligned["close"] / aligned["open"]).where(same_day_ok)
    aligned["raw_ret"] = raw_ret
    aligned["turnover"] = aligned["close"] * aligned["volume"]
    return aligned, issues


def build_panel(frames: Dict[str, pd.DataFrame], benchmark: str, *, origin: str,
                snapshot_id: str) -> Panel:
    if benchmark not in frames or frames[benchmark].empty:
        raise ValueError(f"benchmark {benchmark!r} has no bars; no calendar can be built")
    cleaned: Dict[str, pd.DataFrame] = {}
    issues: Dict[str, List[Dict[str, Any]]] = {}
    for ticker, frame in frames.items():
        if frame is None or frame.empty:
            continue
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        annulled = frame.index.isin(list(CANCELLED_SESSIONS))
        if annulled.any():
            issues.setdefault(ticker, []).extend(
                {"date": day, "issue": "cancelled_session_bar_removed"}
                for day in frame.index[annulled])
            frame = frame[~annulled]
        cleaned[ticker] = frame

    market = cleaned[benchmark]
    calendar = SessionCalendar(market.index[market["close"].notna()])

    # Dates the benchmark lacks but the market evidently traded.
    stocks = [f for t, f in cleaned.items() if t != benchmark]
    gaps: List[str] = []
    if stocks:
        traded: Dict[str, int] = {}
        for frame in stocks:
            active = frame.index[(frame["volume"].fillna(0) > 0) & frame["close"].notna()]
            for day in active:
                if calendar.first < day < calendar.last and not calendar.is_session(day):
                    traded[day] = traded.get(day, 0) + 1
        gaps = sorted(day for day, count in traded.items()
                      if count >= CALENDAR_GAP_SHARE * len(stocks))
    merged = sorted({calendar.on_or_after(day) for day in gaps} - {None})

    bars: Dict[str, pd.DataFrame] = {}
    for ticker, frame in cleaned.items():
        bars[ticker], found = derive(frame, calendar.sessions,
                                     is_benchmark=(ticker == benchmark), merged=merged)
        if found:
            issues.setdefault(ticker, []).extend(found)
    return Panel(calendar, bars, benchmark, origin, snapshot_id, issues,
                 calendar_gaps=gaps, merged_sessions=merged)


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
