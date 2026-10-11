"""From stored raw data to a context the hypotheses can run on.

Two rules are enforced here rather than left to whoever runs a command:

* **No real outcome before the rules.** A real-data run needs the current
  protocol hash to be registered in the database already.
* **No real outcome on a partial sample.** A real-data run needs every anchor
  of the KAP sample frame to have been attempted. An "interim look" on
  whatever has arrived so far is the stock-level version of peeking, and there
  is no flag for it.
* **A gap is counted, and a large one stops the run.** Disclosures the gateway
  refused are not in the sample. Up to the registered tolerance the run
  proceeds and the manifest names each missing disclosure; above it the run
  refuses.

Diagnostics that read no return -- how many events, of which category, with
which tickers resolved -- can be run at any time.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from stock_research import events, guards, limits, store
from stock_research.config import (
    BENCHMARK_TICKER, KAP_FRAME_VERSION, REPOSITORY_ROOT, SEALED_INDEX_BOUNDARY,
)
from stock_research.data import kap, prices
from stock_research.entities import Linker, MENTION_MATERIAL
from stock_research.hypotheses import registry
from stock_research.hypotheses.common import Context

#: Covers the 2023 KAP sample with 130 sessions before its first event and
#: 20 after its last.
PRICE_START, PRICE_END = "2022-06-01", "2024-03-01"
MIN_NEWS_EVENTS = 200


class NotReady(RuntimeError):
    """A real-data run was asked for before its preconditions were met."""


def code_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPOSITORY_ROOT),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                   # pragma: no cover
        return None


# -- Prices ---------------------------------------------------------------------
def ingest_prices(db_path=None, *, snapshot_id: Optional[str] = None,
                  provider: Optional[prices.PriceProvider] = None) -> Dict[str, Any]:
    """Fetch bars for every code among sampled disclosures, plus the benchmark."""

    symbols = [BENCHMARK_TICKER] + events.candidate_symbols(db_path)
    snapshot_id = snapshot_id or f"yahoo-{PRICE_START}-{PRICE_END}"
    counts = prices.ingest(symbols, PRICE_START, PRICE_END, provider or prices.YahooProvider(),
                           snapshot_id, db_path)
    return {"snapshot_id": snapshot_id, "symbols": len(symbols), **counts}


def latest_snapshot(db_path=None) -> Optional[str]:
    with store.connect(db_path) as con:
        row = con.execute(
            "SELECT snapshot_id FROM sr_price_availability GROUP BY snapshot_id "
            "ORDER BY MAX(retrieved_at) DESC LIMIT 1").fetchone()
    return row[0] if row else None


# -- News diagnostics (no outcomes) ----------------------------------------------
def news_diagnostics(index_db: Optional[str | Path] = None, db_path=None) -> Dict[str, Any]:
    """How many scored headlines link to a listed issuer, and how.

    Reads the index database read-only and looks at titles and timestamps
    only. Headlines whose session falls inside the sealed index window are
    counted and set aside; nothing about returns is read for any of them.
    """

    path = Path(index_db or guards.INDEX_DB)
    if not path.exists():
        return {"available": False, "reason": f"{path.name} not found"}
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        frame = pd.read_sql_query(
            "SELECT id, title, signal_date, published_timestamp, sentiment_score "
            "FROM headlines WHERE sentiment_score IS NOT NULL", connection)
    finally:
        connection.close()

    roster: List[str] = []
    try:
        with store.connect(db_path) as con:
            for row in con.execute("SELECT stock_codes FROM sr_raw_members"):
                roster += [c.strip() for c in (row[0] or "").split(",") if c.strip()]
    except sqlite3.OperationalError:
        pass
    linker = Linker(known_tickers=roster)

    sealed = guards.sealed_from(index_db)
    outside = frame if sealed is None else frame[frame["signal_date"] < sealed]
    confirmed = material_single = review = 0
    tickers: Dict[str, int] = {}
    for title in outside["title"]:
        result = linker.link(title)
        review += bool(result.review)
        if not result.links:
            continue
        confirmed += 1
        if len(result.links) == 1 and result.mention_type == MENTION_MATERIAL:
            material_single += 1
            tickers[result.links[0].ticker] = tickers.get(result.links[0].ticker, 0) + 1
    return {
        "scored_headlines": int(len(frame)),
        "sealed_from": sealed,
        "outside_sealed_window": int(len(outside)),
        "with_confirmed_issuer": confirmed,
        "queued_for_review": review,
        "single_issuer_material": material_single,
        "distinct_issuers": len(tickers),
        "with_full_timestamp": int(outside["published_timestamp"].notna().sum()),
        "required_for_stock_level_tests": MIN_NEWS_EVENTS,
    }


# -- Availability -----------------------------------------------------------------
def availability(kap_event_count: int, news: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """What each dataset can support, with the measured reason when it cannot."""

    enough_news = news.get("single_issuer_material", 0) >= MIN_NEWS_EVENTS
    news_reason = (
        f"{news.get('single_issuer_material', 0)} single-issuer material headlines "
        f"across {news.get('distinct_issuers', 0)} issuers lie outside the sealed "
        f"index window; stock-level news tests need at least {MIN_NEWS_EVENTS}, "
        "and no stock price snapshot covers the news period"
        if "single_issuer_material" in news else news.get("reason", "not measured"))
    cap = prices.market_cap_availability()
    intraday = prices.intraday_availability()
    return {
        "kap_events": {"available": kap_event_count > 0,
                       "reason": f"{kap_event_count} sampled disclosures"},
        "news_events": {"available": bool(enough_news), "reason": news_reason},
        "point_in_time_market_cap": {"available": cap.available, "reason": cap.reason},
        "complete_company_news": {
            "available": False,
            "reason": "company news is known only through a systematic sample of "
                      "KAP disclosures and a scraped set of headlines; absence of "
                      "news for a stock on a day cannot be established"},
        "english_news": {"available": False,
                         "reason": "all collected sources are Turkish-language"},
        "intraday_prices": {"available": intraday.available, "reason": intraday.reason},
    }


# -- Contexts ---------------------------------------------------------------------
def real_context(db_path=None, *, snapshot_id: Optional[str] = None,
                 index_db: Optional[str | Path] = None) -> Tuple[Context, Dict[str, Any]]:
    """Assemble the real-data context. Reads prices to build the panel and to
    pick each issuer's most liquid line; computes no abnormal return."""

    snapshot_id = snapshot_id or latest_snapshot(db_path)
    if snapshot_id is None:
        raise NotReady("no price snapshot; run `ingest-prices` first")
    panel = prices.load_panel(snapshot_id, BENCHMARK_TICKER, db_path)
    details = events.load_details(db_path)
    kap_events = events.build_kap_events(details, panel) if len(details) else pd.DataFrame()
    progress = kap.progress(db_path)
    news = news_diagnostics(index_db, db_path)
    ctx = Context(
        panel=panel, kap_events=kap_events,
        availability=availability(len(kap_events), news),
        sealed=guards.sealed_from(index_db),
        frame_complete=bool(progress["frame_complete"]),
    )
    meta = {"snapshot_id": snapshot_id, "frame": progress, "news": news,
            "details": int(len(details)),
            "details_unavailable": int(progress["details_unavailable"])}
    return ctx, meta


def _issue_counts(ctx: Context) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for found in ctx.panel.issues.values():
        for issue in found:
            counts[issue["issue"]] = counts.get(issue["issue"], 0) + 1
    return counts


def _unserved(snapshot_id: str, db_path=None) -> Dict[str, int]:
    """How many requested symbols the provider could not serve. Among them
    are delisted companies, which is where survivorship bias enters."""

    try:
        with store.connect(db_path) as con:
            return {row[0]: row[1] for row in con.execute(
                "SELECT status, COUNT(*) FROM sr_price_availability WHERE snapshot_id = ? "
                "GROUP BY status", (snapshot_id,))}
    except sqlite3.OperationalError:
        return {}


def describe(ctx: Context, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Counts only: what is in the sample, with no return anywhere in it."""

    events_table = ctx.kap_events
    out: Dict[str, Any] = {
        "frame": meta["frame"], "snapshot_id": meta["snapshot_id"],
        "sessions": [ctx.panel.calendar.first, ctx.panel.calendar.last],
        "tickers_with_bars": len(ctx.panel.bars) - 1,
        "tickers_with_price_issues": len(ctx.panel.issues),
        "kap_events": int(len(events_table)), "news": meta["news"],
        "kap_details_the_gateway_could_not_serve": meta.get("details_unavailable", 0),
        "availability": ctx.availability,
        "price_issues": _issue_counts(ctx),
        "calendar_gaps": ctx.panel.calendar_gaps,
        "merged_sessions": ctx.panel.merged_sessions,
        "cancelled_sessions": sorted(prices.CANCELLED_SESSIONS),
        "price_limit_check": limits.empirical_check(ctx.panel),
        "tickers_without_data": _unserved(meta["snapshot_id"]),
    }
    if len(events_table):
        out.update({
            "by_category": events_table["category"].value_counts().to_dict(),
            "by_bucket": events_table["bucket"].value_counts().to_dict(),
            "ticker_resolved": int(events_table["ticker"].notna().sum()),
            "timing_ambiguous": int(events_table["timing_ambiguous"].sum()),
            "updates_or_corrections": int((events_table["is_update"].eq(True)
                                           | events_table["is_correction"].eq(True)).sum()),
            "names_other_issuer": int(events_table["names_other_issuer"].sum()),
            "distinct_day0": int(events_table["day0"].nunique()),
            "unmapped_forms": events_table.loc[
                events_table["category_rule"] == "unmapped_form_name", "subject"
            ].value_counts().head(15).to_dict(),
        })
        insider = events_table[events_table["category"] == events.INSIDER_TRADE]
        if len(insider):
            out["insider"] = {
                "notifications": int(len(insider)),
                "by_status": insider["insider_parse_status"].value_counts().to_dict(),
                "by_side": insider["insider_side"].fillna("none").value_counts().to_dict(),
                "by_kind": insider["insider_kind"].fillna("none").value_counts().to_dict(),
                "purchases_by_venue": insider.loc[
                    insider["insider_side"] == "buy", "insider_venue"
                ].fillna("unknown").value_counts().to_dict(),
            }
    return out


def _digest(rows) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(list(row), ensure_ascii=False, default=str).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def dataset_provenance(db_path=None, snapshot_id: Optional[str] = None) -> Dict[str, Any]:
    """Digests of the data a run reads, and of how the sample was fetched.

    A snapshot id names a set of rows; it does not show the rows are the same
    ones another run saw. These digests do. They are computed over content
    only (no retrieval timestamps), in a fixed order, so the same data gives
    the same digest on any machine.
    """

    snapshot_id = snapshot_id or latest_snapshot(db_path)
    with store.connect(db_path) as con:
        kap_rows = con.execute(
            "SELECT d.disclosure_index, d.payload_sha256, d.published_raw, d.sender_codes, "
            "d.related_stocks, d.subject_tr FROM sr_raw_kap_detail d "
            "JOIN sr_raw_kap_listing l USING (disclosure_index) WHERE l.frame_version = ? "
            "ORDER BY d.disclosure_index", (KAP_FRAME_VERSION,)).fetchall()
        price_rows = con.execute(
            "SELECT ticker, date, open, high, low, close, adj_close, volume, dividend, "
            "split_ratio FROM sr_raw_price_bars WHERE snapshot_id = ? ORDER BY ticker, date",
            (snapshot_id,)).fetchall()
        member_rows = con.execute(
            "SELECT member_id, title, stock_codes, member_type FROM sr_raw_members "
            "ORDER BY snapshot_id, member_id").fetchall()
        failures = con.execute("SELECT COUNT(*) FROM sr_kap_detail_failures").fetchone()[0]
        ingest = [dict(r) for r in con.execute(
            "SELECT run_id, event, at, ingester_sha256, code_commit FROM sr_ingest_log "
            "ORDER BY event_id")]
        retrieved = con.execute(
            "SELECT MIN(retrieved_at), MAX(retrieved_at) FROM sr_raw_price_bars "
            "WHERE snapshot_id = ?", (snapshot_id,)).fetchone()
    return {
        "kap_frame_version": KAP_FRAME_VERSION,
        "kap_sample_digest": _digest(kap_rows),
        "price_snapshot_id": snapshot_id,
        "price_snapshot_digest": _digest(price_rows),
        "price_retrieved_between": [retrieved[0], retrieved[1]],
        "members_snapshot_digest": _digest(member_rows),
        "counts": {"kap_details": len(kap_rows), "price_bars": len(price_rows),
                   "members": len(member_rows), "refused_detail_requests": failures},
        "ingestion_log": ingest,
        "ingester_sha256_now": kap.ingester_sha256(),
    }


def run_real(db_path=None, *, index_db: Optional[str | Path] = None) -> Dict[str, Any]:
    """The registered real-data run. Refuses unless the rules were registered
    first and the sample frame is complete."""

    store.init_db(db_path)
    digest = registry.protocol_hash()
    with store.connect(db_path) as con:
        registered = con.execute(
            "SELECT registered_at FROM sr_protocols WHERE protocol_hash = ?", (digest,)
        ).fetchone()
    if not registered:
        raise NotReady(
            f"protocol {digest[:16]} is not registered; the code has changed since "
            "registration, or `register` was never run")
    ctx, meta = real_context(db_path, index_db=index_db)
    frame = meta["frame"]
    if not frame["ingestion_complete"]:
        raise NotReady(
            f"KAP sample ingestion is incomplete ({frame['blocks_attempted']} of "
            f"{frame['blocks_total']} anchors attempted); no outcome is read on a "
            "partial sample")
    if not frame["within_tolerance"]:
        raise NotReady(
            f"{frame['details_unavailable']} of {frame['targets']} sampled disclosures "
            f"are unavailable ({frame['unavailable_share']:.2%}); the registered "
            f"tolerance is {frame['max_unavailable_share']:.0%}. Rerun `ingest-kap` "
            "to ask for them again")

    results = registry.run_all(ctx)
    manifest = registry.manifest(
        ctx, snapshot_id=meta["snapshot_id"], code_commit=code_commit(),
        analysis_date=datetime.now(timezone.utc).date().isoformat(),
        extra={"kap_frame_version": KAP_FRAME_VERSION,
               "protocol_registered_at": registered[0],
               "kap_sample": {
                   "targets": frame["targets"],
                   "details_stored": frame["details_stored"],
                   "details_unavailable": frame["details_unavailable"],
                   "unavailable_share": frame["unavailable_share"],
                   "unavailable_indices": kap.unavailable_details(db_path),
               },
               "provenance": dataset_provenance(db_path, meta["snapshot_id"]),
               "sample": describe(ctx, meta)})
    registry.store_results(manifest, results, db_path)
    return {"manifest": manifest, "results": results}


def latest_results(db_path=None) -> Optional[Dict[str, Any]]:
    import json

    with store.connect(db_path) as con:
        row = con.execute(
            "SELECT manifest_hash, manifest_json FROM sr_manifests ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        results = {r["hypothesis"]: json.loads(r["result_json"]) for r in con.execute(
            "SELECT hypothesis, result_json FROM sr_results WHERE manifest_hash = ?", (row[0],))}
    return {"manifest": json.loads(row[1]), "results": results}


def synthetic_demo() -> Dict[str, Dict[str, Any]]:
    """Every hypothesis on a synthetic context that contains its effect.

    For showing that the machinery works and what a report looks like. Each
    result is marked synthetic and ``claim_allowed`` is false throughout.
    """

    from stock_research import fixtures

    results: Dict[str, Dict[str, Any]] = {}
    for key, effect in (("H1", "h1"), ("H2", "h2"), ("H3", "h3"), ("H4", "h4")):
        results[key] = registry.RUNNERS[key](fixtures.news_context(effect=effect))
    limit_ctx, kap_ctx = fixtures.limit_context(effect=True), fixtures.kap_context(effect=True)
    for key in ("H5", "H6"):
        results[key] = registry.RUNNERS[key](limit_ctx)
    for key in ("H7", "H8"):
        results[key] = registry.RUNNERS[key](kap_ctx)
    results["H9"] = registry.RUNNERS["H9"](fixtures.language_context(effect=True))

    from stock_research import stats
    from stock_research.hypotheses import common

    order = list(results)
    raw = [(results[k]["primary"] or {}).get("p") if results[k]["status"] is None else None
           for k in order]
    for key, p_holm in zip(order, stats.holm(raw, family_size=common.FAMILY_SIZE)):
        if results[key]["status"] is None:
            results[key]["primary"]["p_holm"] = p_holm
            results[key]["status"] = common.decide(registry.SPECS[key], results[key]["primary"])
    return results
