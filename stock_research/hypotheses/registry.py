"""The registered protocol: nine hypotheses, one hash, one verdict rule.

``protocol_document`` collects everything that can change a result -- the nine
specifications, every rule version, every constant -- and ``protocol_hash``
fingerprints it. A result is only comparable to another result with the same
hash. Registering stores the document append-only; the commit that contains
this file is the timestamp that says the rules came before the outcomes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from stock_research import config, costs, entities, events, eventstudy, insider, limits, stats, store
from stock_research import calendar as session_calendar
from stock_research.data import prices, secondary
from stock_research.hypotheses import common, h1, h2, h3, h4, h5, h6, h7, h8, h9
from stock_research.hypotheses.common import Context

PROTOCOL_VERSION = "stock-research-protocol-v3"

#: The protocol this one replaces. It stays registered, and anything stored
#: under it stays stored; v3 is a different study, not an edit of v2. v2 was
#: never run: the readiness audit of 2026-10-11 returned BLOCKED before any
#: outcome was read (docs/stock_research/readiness/READINESS_DECISION.md).
SUPERSEDES = {
    "protocol_version": "stock-research-protocol-v2",
    "protocol_hash": "09e9382e77ec44bc06aaaf5c4bde0d64e06a6584d2aeb4b65ed35da0b2e7d99a",
    "registered_at": "2026-10-11T08:39:14+00:00",
    "code_commit": "df6be00",
    "real_results_under_it": 0,
}
#: Every protocol before this one, oldest first. None has a stored result.
LINEAGE = (
    {"protocol_version": "stock-research-protocol-v1",
     "protocol_hash": "92366aa0a6342e9c78c02700a7253b45e5dc4ce0cac6476467057164cb66dfb8",
     "registered_at": "2026-10-11T00:15:04+00:00", "code_commit": "ecdc532"},
    dict(SUPERSEDES),
)
CHANGES_FROM_PREVIOUS = (
    "H7's joint p-value comes from a wild cluster bootstrap (null imposed, "
    "Rademacher weights by day 0, 9,999 draws, fixed seed) of the two-way "
    "clustered Wald statistic. v2 referred that statistic to F(q, G-1), which "
    "rejected a true null about 10% of the time at the 5% level and 3% at the "
    "Holm level on the projected sample.",
    "H7 tests named classes of filing against a reference group of procedural "
    "filings, one pre-specified contrast each. v2 tested whichever classes "
    "reached 30 events against the largest one; most of those were residual "
    "classes with no stated content.",
    "A class is tested only with 30 events on 20 dates from 20 issuers, the "
    "reference group with 60 on 30 from 30. v2 required 30 events on 15 dates "
    "and had no issuer rule.",
    "An issuer's H7 events are thinned so that no two outcome windows overlap. "
    "v2 kept them all and reported thinning as a sensitivity.",
    "H7's reported size is the largest contrast among those individually "
    "significant after Holm adjustment, with a bias-corrected root-mean-square "
    "contrast beside it. v2 reported the highest class mean minus the lowest, "
    "which exceeds the 1% material effect in 85% of samples with no effect.",
    "The sample is a census of the 2023 range, not 80 anchors of 50. Rows "
    "stored under the anchor frame are members of the census and are reused.",
    "A return enters the study only if a second provider's adjusted return "
    "agrees with it within one percentage point; otherwise it is missing. On "
    "corporate-action dates the adjusted return is tested against the price "
    "limit as well.",
    "The 2023 exchange schedule is recorded: two half days, the holidays, the "
    "cancelled session and the closure after it. v2 inferred half days. "
    "Filings dated to the session that reopened the market after the "
    "unscheduled closure are excluded from H7.",
    "Complete retrieval of target disclosures is required. A disclosure may be "
    "accepted as unavailable only after three refusals on three separate days, "
    "up to 0.1% of targets; v2 allowed 1% after one attempt. Every stored "
    "detail must have a parseable timestamp and a subject.",
    "The run refuses unless prices were requested from both sources for every "
    "sampled issuer code and the sessions in the data match the recorded "
    "schedule. The gates are carried by the real-data context, and no "
    "hypothesis computes an outcome on a context whose gates have not passed; "
    "in v2 the checks sat in one function the runners could be called around.",
    "Constants read from the repository-root config (holiday lists, the "
    "economic calendar, the KAP base URL) are digested into the protocol, and "
    "library versions are recorded in every manifest.",
)

#: Files that cannot change a registered result: fixtures and the report
#: renderer only display, the CLI only dispatches, and registry_doc only
#: prints the specifications. Everything else under stock_research/ is
#: fingerprinted, the ingester included.
_NOT_RESULT_BEARING = {"fixtures.py", "reporting.py", "cli.py", "registry_doc.py"}

MODULES = (h1, h2, h3, h4, h5, h6, h7, h8, h9)
SPECS = {module.SPEC.id: module.SPEC for module in MODULES}
RUNNERS: Dict[str, Callable[[Context], Dict[str, Any]]] = {
    module.SPEC.id: module.run for module in MODULES
}

VERDICT_RULE = (
    "data_insufficient if a registered requirement or sample-size rule fails; "
    "supported if the primary estimate has the hypothesised sign, is at least "
    "the material effect, and its Holm-adjusted p-value (family of nine) is "
    "below alpha; execution_not_verifiable in place of supported where the "
    "claim depends on an unverifiable fill; not_supported if the 95% interval "
    "in the hypothesised direction lies below the material effect; otherwise "
    "inconclusive. A joint test can only be supported or inconclusive."
)


def code_fingerprint() -> Dict[str, str]:
    """SHA-256 of every source file that can change a result.

    Constants and specifications are hashed directly, but a rule can also
    change inside a function body. Line endings are normalised first, so the
    fingerprint is the same on a Windows and a Unix checkout.
    """

    root = Path(config.__file__).resolve().parent
    out = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if relative in _NOT_RESULT_BEARING:
            continue
        text = path.read_bytes().replace(b"\r\n", b"\n")
        out[relative] = hashlib.sha256(text).hexdigest()
    return out


def external_constants() -> Dict[str, str]:
    """Digests of the constants fingerprinted modules read from outside the
    package: ``calendar`` reads the holiday lists, ``h4`` the economic
    calendar and ``data.kap`` the gateway's address, all from the
    repository-root ``config``. A change to any of them changes a result
    without changing a line under ``stock_research/``."""

    import config as root

    def digest(value: Any) -> str:
        return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()

    return {"BIST_HOLIDAYS": digest(root.BIST_HOLIDAYS),
            "BIST_HALF_DAYS": digest(root.BIST_HALF_DAYS),
            "ECONOMIC_CALENDAR": digest(root.ECONOMIC_CALENDAR),
            "KAP_BASE_URL": digest(root.KAP_BASE_URL)}


def protocol_document() -> Dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "supersedes": dict(SUPERSEDES),
        "lineage": [dict(entry) for entry in LINEAGE],
        "external_constants": external_constants(),
        "changes_from_previous": list(CHANGES_FROM_PREVIOUS),
        "code_fingerprint": code_fingerprint(),
        "family_size": common.FAMILY_SIZE,
        "alpha": config.ALPHA,
        "verdict_rule": VERDICT_RULE,
        "claim_rule": "a verdict may be reported as a finding only from real "
                      "historical data on a complete sample frame",
        "hypotheses": {key: spec.as_dict() for key, spec in SPECS.items()},
        "versions": {
            "engine": eventstudy.ENGINE_VERSION,
            "calendar": session_calendar.CALENDAR_RULE_VERSION,
            "price_rules": prices.PRICE_RULE_VERSION,
            "taxonomy": events.TAXONOMY_VERSION,
            "entity_linker": entities.LINKER_VERSION,
            "insider_parser": insider.INSIDER_PARSER_VERSION,
            "limit_rule": limits.LIMIT_RULE_VERSION,
            "cost_model": costs.COST_MODEL_VERSION,
            "kap_frame": config.KAP_FRAME_VERSION,
            "schema": config.SCHEMA_VERSION,
        },
        "constants": {
            "event_windows": [list(w) for w in config.EVENT_WINDOWS],
            "estimation_window": list(config.ESTIMATION_WINDOW),
            "estimation_min_observations": config.ESTIMATION_MIN_OBSERVATIONS,
            "benchmark": config.BENCHMARK_TICKER,
            "min_events_per_group": config.MIN_EVENTS_PER_GROUP,
            "min_event_dates_per_group": config.MIN_EVENT_DATES_PER_GROUP,
            "min_clusters_for_inference": stats.MIN_CLUSTERS,
            "jump_threshold": prices.JUMP_THRESHOLD,
            "cancelled_sessions": prices.CANCELLED_SESSIONS,
            "calendar_gap_share": prices.CALENDAR_GAP_SHARE,
            "limit_rules": limits.LIMIT_RULES,
            "limit_tolerance": limits.TOLERANCE,
            "round_trip_cost_scenarios": list(config.ROUND_TRIP_COST_SCENARIOS),
            "kap_frame": {
                "kind": "census",
                "first_index": config.KAP_FRAME_FIRST_INDEX,
                "last_index": config.KAP_FRAME_LAST_INDEX,
                "listing_page": config.KAP_LISTING_PAGE,
                "earlier_frames_reused": list(config.KAP_EARLIER_FRAMES),
                "listing_types": list(config.KAP_LISTING_TYPES),
                "target_classes": list(config.KAP_TARGET_CLASSES),
                "listed_member_mark": config.KAP_LISTED_MEMBER_MARK,
                "max_detail_attempts": config.KAP_MAX_DETAIL_ATTEMPTS,
                "min_attempt_days": config.KAP_MIN_ATTEMPT_DAYS,
                "max_unavailable_share": config.KAP_MAX_UNAVAILABLE_SHARE,
            },
            "h7": {
                "min_events": config.H7_MIN_EVENTS, "min_dates": config.H7_MIN_DATES,
                "min_issuers": config.H7_MIN_ISSUERS,
                "reference_min_events": config.H7_REFERENCE_MIN_EVENTS,
                "reference_min_dates": config.H7_REFERENCE_MIN_DATES,
                "reference_min_issuers": config.H7_REFERENCE_MIN_ISSUERS,
                "bootstrap_draws": config.WILD_BOOTSTRAP_DRAWS,
                "bootstrap_seed": config.WILD_BOOTSTRAP_SEED,
            },
            "reconcile_tolerance": config.RECONCILE_TOLERANCE,
            "action_limit_tolerance": prices.ACTION_LIMIT_TOLERANCE,
            "second_price_source": secondary.SOURCE,
            "exchange_schedule": session_calendar.EXCHANGE_SCHEDULE,
            "sealed_index_boundary": config.SEALED_INDEX_BOUNDARY,
            "subject_categories": events.SUBJECT_CATEGORIES,
            "general_summary_rules": [[c, list(p)] for c, p in events.GENERAL_SUMMARY_RULES],
        },
    }


def canonical(document: Dict[str, Any]) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=str)


def protocol_hash(document: Optional[Dict[str, Any]] = None) -> str:
    return hashlib.sha256(canonical(document or protocol_document()).encode("utf-8")).hexdigest()


def register(db_path=None) -> Dict[str, Any]:
    """Store the current protocol. Idempotent by hash."""

    store.init_db(db_path)
    document = protocol_document()
    digest = protocol_hash(document)
    with store.connect(db_path) as con:
        existing = con.execute(
            "SELECT registered_at FROM sr_protocols WHERE protocol_hash = ?", (digest,)
        ).fetchone()
        if existing:
            return {"protocol_hash": digest, "already_registered": True,
                    "registered_at": existing[0]}
        now = store.now_iso()
        con.execute("INSERT INTO sr_protocols VALUES (?,?,?,?)",
                    (digest, PROTOCOL_VERSION, canonical(document), now))
    return {"protocol_hash": digest, "already_registered": False, "registered_at": now}


def run_all(ctx: Context) -> Dict[str, Dict[str, Any]]:
    """Run every hypothesis, then adjust and decide.

    The Holm family is the nine registered hypotheses. One that cannot run
    contributes no p-value but still counts toward the family size, so thin
    data never makes the correction lighter.
    """

    results = {key: runner(ctx) for key, runner in RUNNERS.items()}
    order = list(results)
    raw = [(results[k]["primary"] or {}).get("p") if results[k]["status"] is None else None
           for k in order]
    adjusted = stats.holm(raw, family_size=common.FAMILY_SIZE)
    for key, p_holm in zip(order, adjusted):
        result = results[key]
        if result["status"] is not None:          # already data_insufficient
            continue
        result["primary"]["p_holm"] = p_holm
        result["status"] = common.decide(SPECS[key], result["primary"])
    return results


def manifest(ctx: Context, *, snapshot_id: str, code_commit: Optional[str],
             analysis_date: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """What a set of results was computed from."""

    document = {
        "protocol_hash": protocol_hash(),
        "protocol_version": PROTOCOL_VERSION,
        "dataset_snapshot": snapshot_id,
        "data_origin": ctx.origin,
        "frame_complete": ctx.frame_complete,
        "code_commit": code_commit,
        "analysis_date": analysis_date,
        "benchmark": config.BENCHMARK_TICKER,
        "return_adjustment": "provider-adjusted close for multi-day returns; "
                             "unadjusted open/close for same-day quantities",
        "sentiment_model": "stored scores from the index pipeline (model and "
                           "prompt version recorded per headline)",
        "bootstrap_seed": config.BOOTSTRAP_SEED,
        "sealed_from": ctx.sealed,
        "availability": ctx.availability,
        **(extra or {}),
    }
    document["manifest_hash"] = hashlib.sha256(canonical(document).encode("utf-8")).hexdigest()
    return document


def store_results(manifest_doc: Dict[str, Any], results: Dict[str, Dict[str, Any]],
                  db_path=None) -> None:
    """Append results under their manifest. A manifest is written once; running
    the same manifest again changes nothing, and nothing is overwritten."""

    store.init_db(db_path)
    digest = manifest_doc["manifest_hash"]
    with store.connect(db_path) as con:
        if con.execute("SELECT 1 FROM sr_manifests WHERE manifest_hash = ?", (digest,)).fetchone():
            return
        now = store.now_iso()
        con.execute("INSERT INTO sr_manifests VALUES (?,?,?)", (digest, canonical(manifest_doc), now))
        for key, result in results.items():
            con.execute("INSERT INTO sr_results VALUES (?,?,?,?,?)",
                        (digest, key, result["status"], canonical(result), now))
