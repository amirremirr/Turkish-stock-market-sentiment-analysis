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
from typing import Any, Callable, Dict, List, Optional, Tuple

from stock_research import config, costs, entities, events, eventstudy, insider, limits, stats, store
from stock_research import calendar as session_calendar
from stock_research.data import prices
from stock_research.hypotheses import common, h1, h2, h3, h4, h5, h6, h7, h8, h9
from stock_research.hypotheses.common import Context

PROTOCOL_VERSION = "stock-research-protocol-v1"

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


def protocol_document() -> Dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
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
            "limit_rules": limits.LIMIT_RULES,
            "limit_tolerance": limits.TOLERANCE,
            "round_trip_cost_scenarios": list(config.ROUND_TRIP_COST_SCENARIOS),
            "kap_frame": {
                "first_index": config.KAP_FRAME_FIRST_INDEX,
                "last_index": config.KAP_FRAME_LAST_INDEX,
                "blocks": config.KAP_FRAME_BLOCKS,
                "listing_types": list(config.KAP_LISTING_TYPES),
                "target_classes": list(config.KAP_TARGET_CLASSES),
            },
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
