"""Command line for the stock-level study.

    python -m stock_research.cli status           what is stored, what is registered
    python -m stock_research.cli ingest-kap       resume the KAP sample (slow: 6 calls/minute)
    python -m stock_research.cli ingest-prices    fetch bars for sampled issuers
    python -m stock_research.cli describe         sample counts; reads no return
    python -m stock_research.cli register         register the current protocol
    python -m stock_research.cli run              the registered real-data run (refuses if not ready)
    python -m stock_research.cli demo             every hypothesis on synthetic data
    python -m stock_research.cli report           HTML report from the latest stored real run
    python -m stock_research.cli evaluate-linker  score the entity linker on the labelled set
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from stock_research import pipeline, reporting, store
from stock_research.config import REPOSITORY_ROOT
from stock_research.data import kap
from stock_research.hypotheses import registry


def _print(value) -> None:
    print(json.dumps(value, indent=1, default=str, ensure_ascii=False))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=[
        "status", "ingest-kap", "ingest-prices", "describe", "register", "run", "demo",
        "report", "evaluate-linker"])
    parser.add_argument("--db")
    parser.add_argument("--max-hours", type=float, default=8.0)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    store.init_db(args.db)

    if args.command == "status":
        digest = registry.protocol_hash()
        with store.connect(args.db) as con:
            registered = con.execute(
                "SELECT registered_at FROM sr_protocols WHERE protocol_hash = ?", (digest,)).fetchone()
            runs = con.execute("SELECT COUNT(*) FROM sr_manifests").fetchone()[0]
        _print({"protocol_hash": digest, "registered_at": registered[0] if registered else None,
                "kap_frame": kap.progress(args.db), "tables": store.table_counts(args.db),
                "price_snapshot": pipeline.latest_snapshot(args.db), "stored_runs": runs})
    elif args.command == "ingest-kap":
        _print(kap.run(args.db, max_hours=args.max_hours))
    elif args.command == "ingest-prices":
        _print(pipeline.ingest_prices(args.db))
    elif args.command == "describe":
        ctx, meta = pipeline.real_context(args.db)
        _print(pipeline.describe(ctx, meta))
    elif args.command == "register":
        _print(registry.register(args.db))
    elif args.command == "run":
        try:
            outcome = pipeline.run_real(args.db)
        except pipeline.NotReady as refusal:
            print(f"REFUSED  {refusal}")
            return 3
        files = reporting.write_report(outcome["results"], manifest=outcome["manifest"],
                                       overview=outcome["manifest"].get("sample"), name="report")
        _print({key: {"status": r["status"], "reasons": r["sufficiency"]["reasons"]}
                for key, r in outcome["results"].items()})
        _print(files)
    elif args.command == "demo":
        results = pipeline.synthetic_demo()
        files = reporting.write_report(
            results, name="demo_synthetic",
            overview={"note": "synthetic data with planted effects; not a finding"})
        _print({key: r["status"] for key, r in results.items()})
        _print(files)
    elif args.command == "report":
        latest = pipeline.latest_results(args.db)
        if latest is None:
            print("no stored real-data run; use `run` when the sample is ready, or `demo`")
            return 3
        _print(reporting.write_report(latest["results"], manifest=latest["manifest"],
                                      overview=latest["manifest"].get("sample"), name="report"))
    elif args.command == "evaluate-linker":
        from stock_research.entities import evaluate

        path = REPOSITORY_ROOT / "tests" / "stock_research" / "entity_eval.csv"
        with open(path, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="|"))
        _print(evaluate([{"text": r["text"], "tickers": [t for t in r["tickers"].split(";") if t]}
                         for r in rows]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
