"""Synthetic panels and contexts with known answers.

Used for two things only: testing that an estimator recovers an effect that
was put there (and finds nothing where nothing was put), and letting the
report render when real data are absent. Everything built here carries
``origin = "synthetic"`` and nothing derived from it may be reported as a
finding.

Returns are built from an overnight part and an intraday part that are drawn
independently, so the opening gap carries no information about the rest of the
day unless a test puts it there.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from stock_research.data.prices import ORIGIN_SYNTHETIC, Panel, build_panel
from stock_research.hypotheses.common import Context

BENCHMARK = "MKT"
LIMIT_MOVE = 0.0995
Effects = Dict[Tuple[str, int], float]            # (ticker, session index) -> return


def weekdays(start: str, count: int, *, skip: Sequence[str] = ()) -> List[str]:
    day, out, skipped = date.fromisoformat(start), [], set(skip)
    while len(out) < count:
        if day.weekday() < 5 and day.isoformat() not in skipped:
            out.append(day.isoformat())
        day += timedelta(days=1)
    return out


def _bars(days: Sequence[str], overnight: np.ndarray, intraday: np.ndarray, *,
          start_price: float, rng: np.random.Generator,
          forced: Optional[Dict[int, str]] = None) -> pd.DataFrame:
    """OHLCV where ``close_t = close_{t-1} * (1 + overnight_t + intraday_t)``
    and ``open_t = close_{t-1} * (1 + overnight_t)``."""

    total = overnight + intraday
    close = start_price * np.cumprod(1 + total)
    prior = np.concatenate([[start_price], close[:-1]])
    open_ = prior * (1 + overnight)
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    for index, side in (forced or {}).items():
        if side == "up":
            high[index], low[index] = close[index], open_[index]
        else:
            low[index], high[index] = close[index], open_[index]
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "adj_close": close,
        "volume": rng.integers(200_000, 2_000_000, size=len(days)).astype(float),
        "dividend": 0.0, "split_ratio": 0.0,
    }, index=list(days))


def make_panel(days: Sequence[str], tickers: Sequence[str], *, seed: int,
               intraday_effects: Optional[Effects] = None,
               overnight_effects: Optional[Effects] = None,
               forced: Optional[Dict[Tuple[str, int], str]] = None,
               sector_sd: float = 0.0, sectors: int = 4) -> Panel:
    """A one-factor panel. Effects add to the named part of the return on the
    given session; ``forced`` pins a session to a limit move with the close at
    the day's extreme."""

    rng = np.random.default_rng(seed)
    n = len(days)
    market_o, market_d = rng.normal(0, 0.006, n), rng.normal(0.0003, 0.010, n)
    sector = rng.normal(0, sector_sd, (sectors, n)) if sector_sd else None
    frames = {BENCHMARK: _bars(days, market_o, market_d, start_price=1000.0, rng=rng)}
    for number, name in enumerate(tickers):
        beta = rng.uniform(0.6, 1.4)
        overnight = beta * market_o + rng.normal(0, 0.010, n)
        intraday = beta * market_d + rng.normal(0, 0.017, n)
        if sector is not None:
            intraday = intraday + sector[number % sectors]
        for (ticker, index), value in (intraday_effects or {}).items():
            if ticker == name and 0 <= index < n:
                intraday[index] += value
        for (ticker, index), value in (overnight_effects or {}).items():
            if ticker == name and 0 <= index < n:
                overnight[index] += value
        pinned = {index: side for (ticker, index), side in (forced or {}).items()
                  if ticker == name and 0 < index < n}
        for index, side in pinned.items():
            overnight[index] = 0.0
            intraday[index] = LIMIT_MOVE if side == "up" else -LIMIT_MOVE
        frames[name] = _bars(days, overnight, intraday, start_price=10.0, rng=rng, forced=pinned)
    return build_panel(frames, BENCHMARK, origin=ORIGIN_SYNTHETIC,
                       snapshot_id=f"synthetic-seed{seed}")


def synthetic_panel(*, tickers: int = 40, sessions: int = 420, seed: int = 1,
                    start: str = "2022-01-03",
                    effects: Optional[Dict[Tuple[str, str], Dict[int, float]]] = None,
                    sector_sd: float = 0.0) -> Panel:
    """Convenience wrapper: ``effects`` maps ``(ticker, day0)`` to
    ``{relative_day: abnormal_return}``."""

    days = weekdays(start, sessions)
    position = {d: i for i, d in enumerate(days)}
    flat: Effects = {}
    for (ticker, day0), path in (effects or {}).items():
        for relative, value in path.items():
            key = (ticker, position[day0] + relative)
            flat[key] = flat.get(key, 0.0) + value
    return make_panel(days, [f"S{n:03d}" for n in range(tickers)], seed=seed,
                      intraday_effects=flat, sector_sd=sector_sd)


def random_events(panel: Panel, count: int, *, seed: int = 2, first: int = 150,
                  last_margin: int = 25, per_day: int = 1) -> List[Dict[str, str]]:
    """``count`` events on random tickers. ``per_day`` > 1 stacks several
    events on each chosen session, which is what makes clustering matter."""

    rng = np.random.default_rng(seed)
    names = [t for t in panel.bars if t != panel.benchmark]
    sessions = panel.calendar.sessions
    events: List[Dict[str, str]] = []
    while len(events) < count:
        day = sessions[int(rng.integers(first, len(sessions) - last_margin))]
        for ticker in rng.choice(names, size=min(per_day, len(names)), replace=False):
            if len(events) < count:
                events.append({"event_id": f"E{len(events):05d}", "ticker": str(ticker),
                               "day0": day})
    return events


# -- Contexts ------------------------------------------------------------------
ALL_AVAILABLE = {key: {"available": True, "reason": "synthetic"} for key in (
    "news_events", "kap_events", "point_in_time_market_cap", "complete_company_news",
    "english_news", "intraday_prices")}


def _spread(effects: Effects, ticker: str, index: int, days: Sequence[int], total: float) -> None:
    for relative in days:
        key = (ticker, index + relative)
        effects[key] = effects.get(key, 0.0) + total / len(days)


def news_context(*, effect: Optional[str] = None, seed: int = 11, sessions: int = 520,
                 news_tickers: int = 24, events_per_ticker: int = 42,
                 availability: Optional[Dict[str, Any]] = None) -> Context:
    """Stock-level news with sentiment, and optionally one planted effect.

    ``effect`` is ``None`` (no relation between news and returns) or one of:

    ``"h1"``  signed drift over days +2..+5, small caps only
    ``"h2"``  after-close news moves the open; a third of the gap continues intraday
    ``"h3"``  CAR(0,+5) follows the *surprise* against the ticker's own level
    ``"h4"``  heavier-than-usual coverage is followed by lower CAR(+1,+10)
    """

    rng = np.random.default_rng(seed)
    days = weekdays("2022-01-03", sessions)
    tickers = [f"S{n:03d}" for n in range(60)]
    # Spread across the size range so every tercile has covered stocks.
    covered = [tickers[int(i * len(tickers) / news_tickers)] for i in range(news_tickers)]
    # For the surprise test the ticker's own level must dominate its sentiment,
    # otherwise the raw level and the surprise are nearly the same variable.
    level_width, noise_sd = (0.8, 0.15) if effect == "h3" else (0.45, 0.35)
    if effect == "h3":
        events_per_ticker = max(events_per_ticker, 64)
    cap = {t: float(np.exp(4 + 6 * i / len(tickers))) for i, t in enumerate(tickers)}
    tercile = {t: ("small" if i < 20 else "mid" if i < 40 else "large")
               for i, t in enumerate(tickers)}
    level = {t: rng.uniform(-level_width, level_width) for t in covered}

    intraday: Effects = {}
    overnight: Effects = {}
    rows: List[Dict[str, Any]] = []
    story = 0
    for ticker in covered:
        chosen = sorted(rng.choice(np.arange(150, sessions - 25), size=events_per_ticker,
                                   replace=False))
        for index in chosen:
            sentiment = float(np.clip(level[ticker] + rng.normal(0, noise_sd), -1, 1))
            stories = 1 + int(rng.poisson(0.6)) + (int(rng.integers(3, 7)) if rng.random() < 0.12 else 0)
            after_close = rng.random() < 0.6
            if effect == "h1":
                size_effect = {"small": 0.016, "mid": 0.006, "large": 0.0}[tercile[ticker]]
                _spread(intraday, ticker, index, (2, 3, 4, 5), np.sign(sentiment) * size_effect)
            elif effect == "h2" and after_close:
                gap = 0.03 * sentiment + rng.normal(0, 0.01)
                overnight[(ticker, index)] = overnight.get((ticker, index), 0.0) + gap
                intraday[(ticker, index)] = intraday.get((ticker, index), 0.0) + gap / 3
            elif effect == "h3":
                _spread(intraday, ticker, index, (0, 1, 2, 3, 4, 5),
                        0.25 * (sentiment - level[ticker]))
            elif effect == "h4":
                _spread(intraday, ticker, index, tuple(range(1, 11)), -0.02 * np.log(stories))
            for _ in range(stories):
                rows.append({
                    "event_id": f"N{story:06d}", "story_id": f"N{story:06d}",
                    "ticker": ticker, "subject": ticker, "day0": days[index],
                    "bucket": "post_close" if after_close else "pre_open",
                    "timing_ambiguous": False, "sentiment": sentiment, "n_articles": 1,
                    "mention_type": "material", "n_issuers": 1, "link_confirmed": True,
                    "first_published_utc": f"{days[index - 1]}T17:00:00+00:00",
                    "language": "tr",
                })
                story += 1

    panel = make_panel(days, tickers, seed=seed + 1, intraday_effects=intraday,
                       overnight_effects=overnight)
    size = pd.DataFrame([{"ticker": t, "date": days[i], "market_cap": cap[t]}
                         for i in range(0, sessions, 20) for t in tickers])
    return Context(panel=panel, news_events=pd.DataFrame(rows), size=size,
                   availability=dict(availability or ALL_AVAILABLE))


def kap_context(*, effect: bool = False, seed: int = 21, sessions: int = 520,
                per_category: int = 70, insider_buys: int = 90) -> Context:
    """Categorised disclosures. With ``effect``: ``cat_up`` gains 3% over
    days 0..+5, ``cat_down`` loses 2%, and insider purchases gain 4% over
    days +1..+20."""

    rng = np.random.default_rng(seed)
    days = weekdays("2022-01-03", sessions)
    tickers = [f"S{n:03d}" for n in range(60)]
    intraday: Effects = {}
    rows: List[Dict[str, Any]] = []
    taken = set()

    def place() -> Tuple[str, int]:
        while True:
            key = (str(rng.choice(tickers)), int(rng.integers(150, sessions - 25)))
            if key not in taken:
                taken.add(key)
                return key

    def row(category: str, ticker: str, index: int, **extra: Any) -> None:
        rows.append({
            "event_id": f"K{len(rows):05d}", "ticker": ticker, "day0": days[index],
            "bucket": str(rng.choice(["post_close", "pre_open", "during_session"])),
            "timing_ambiguous": False, "category": category, "is_update": False,
            "is_correction": False, "insider_parse_status": None, "insider_side": None,
            "insider_value": None, "insider_role": None, **extra})

    for category, total in (("cat_up", 0.03), ("cat_down", -0.02), ("cat_flat_a", 0.0),
                            ("cat_flat_b", 0.0), ("cat_flat_c", 0.0)):
        for _ in range(per_category):
            ticker, index = place()
            if effect and total:
                _spread(intraday, ticker, index, (0, 1, 2, 3, 4, 5), total)
            row(category, ticker, index)
    for _ in range(insider_buys):
        ticker, index = place()
        if effect:
            _spread(intraday, ticker, index, tuple(range(1, 21)), 0.04)
        row("insider_trade", ticker, index, insider_parse_status="parsed",
            insider_side="buy", insider_value=float(rng.uniform(1e5, 5e6)),
            insider_role="board_member")
    for _ in range(25):
        ticker, index = place()
        row("insider_trade", ticker, index, insider_parse_status="parsed", insider_side="sell")

    panel = make_panel(days, tickers, seed=seed + 1, intraday_effects=intraday)
    availability = dict(ALL_AVAILABLE)
    availability["point_in_time_market_cap"] = {"available": False, "reason": "not in this fixture"}
    return Context(panel=panel, kap_events=pd.DataFrame(rows), availability=availability)


def limit_context(*, effect: bool = False, seed: int = 31, sessions: int = 460,
                  limit_downs: int = 240, streak_starts: int = 260) -> Context:
    """Limit-down closes with and without news, and limit-up streaks.

    With ``effect``: a no-news limit-down gains 4% over days +1..+3, and a
    limit-up streak continues with probability 0.8 on days with company news
    against 0.3 without. Otherwise there is no reversal and continuation is
    0.55 regardless of news.
    """

    rng = np.random.default_rng(seed)
    days = weekdays("2022-01-03", sessions)
    tickers = [f"S{n:03d}" for n in range(70)]
    forced: Dict[Tuple[str, int], str] = {}
    intraday: Effects = {}
    news: List[Dict[str, Any]] = []
    busy: Dict[str, set] = {t: set() for t in tickers}

    def free(ticker: str, start: int, length: int) -> bool:
        return not any(i in busy[ticker] for i in range(start - 2, start + length + 2))

    def add_news(ticker: str, index: int, sentiment: float) -> None:
        news.append({"event_id": f"L{len(news):05d}", "story_id": f"L{len(news):05d}",
                     "ticker": ticker, "subject": ticker, "day0": days[index],
                     "bucket": "pre_open", "timing_ambiguous": False, "sentiment": sentiment,
                     "n_articles": 1, "mention_type": "material", "n_issuers": 1,
                     "link_confirmed": True,
                     "first_published_utc": f"{days[index]}T05:00:00+00:00", "language": "tr"})

    placed = 0
    while placed < limit_downs:
        ticker, index = str(rng.choice(tickers)), int(rng.integers(150, sessions - 30))
        if not free(ticker, index, 5):
            continue
        busy[ticker].update(range(index, index + 5))
        forced[(ticker, index)] = "down"
        if rng.random() < 0.5:
            add_news(ticker, index, -0.6)
        elif effect:
            _spread(intraday, ticker, index, (1, 2, 3), 0.04)
        placed += 1

    placed = 0
    while placed < streak_starts:
        ticker, index = str(rng.choice(tickers)), int(rng.integers(150, sessions - 30))
        if not free(ticker, index, 12):
            continue
        length = 0
        while length < 10:
            forced[(ticker, index + length)] = "up"
            has_news = rng.random() < 0.5
            if has_news:
                add_news(ticker, index + length, 0.6)
            length += 1
            stay = (0.8 if has_news else 0.3) if effect else 0.55
            if rng.random() >= stay:
                break
        busy[ticker].update(range(index, index + length + 1))
        placed += 1

    panel = make_panel(days, tickers, seed=seed + 1, intraday_effects=intraday, forced=forced)
    return Context(panel=panel, news_events=pd.DataFrame(news),
                   availability=dict(ALL_AVAILABLE))


def language_context(*, effect: bool = False, seed: int = 41, events: int = 260) -> Context:
    """The same events reported in English and Turkish. With ``effect`` the
    English report leads by three hours on average; otherwise by zero."""

    rng = np.random.default_rng(seed)
    days = weekdays("2023-01-02", 200)
    base = datetime(2023, 1, 2, 9, tzinfo=timezone.utc)
    english, turkish = [], []
    for number in range(events):
        subject = f"S{number % 40:03d}"
        moment = base + timedelta(days=int(rng.integers(0, 270)), hours=float(rng.uniform(0, 10)),
                                  minutes=number)
        lag = rng.normal(3.0 if effect else 0.0, 6.0)
        sentiment = float(rng.uniform(-1, 1))
        turkish.append({"story_id": f"T{number}", "event_id": f"T{number}", "subject": subject,
                        "ticker": subject, "first_published_utc": moment.isoformat(),
                        "sentiment": sentiment})
        english.append({"story_id": f"E{number}", "subject": subject,
                        "first_published_utc": (moment - timedelta(hours=lag)).isoformat(),
                        "sentiment": sentiment + rng.normal(0, 0.2)})
    panel = make_panel(days, ["S000", "S001"], seed=seed)
    return Context(panel=panel, news_events=pd.DataFrame(turkish),
                   english_events=pd.DataFrame(english), availability=dict(ALL_AVAILABLE))
