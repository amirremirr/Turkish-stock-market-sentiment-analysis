"""Start or resume the v3 census from this worktree, against an explicit database.

    python -m v3_validation.run_census --db <path> --env <path-to-.env> [--max-hours 48]
    python -m v3_validation.run_census --db <path> --check        # preconditions only

The worktree has no credentials of its own. The two gateway credentials are
read from the given ``.env`` into this process's environment and nowhere else:
they are not printed, logged, copied or written. The database path must be
given and is verified before anything is written to it.

Refuses unless: the database holds the registered v1 and v2 protocols and no
stored result (it is the research database, not a scratch file); the anchor
frame is complete with every target detail stored; and no other ingestion
process is running (the gateway's rate limit is shared).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from stock_research import store
from stock_research.config import KAP_ANCHOR_FRAME_BLOCKS, KAP_ANCHOR_FRAME_VERSION, REPOSITORY_ROOT
from stock_research.data import kap
from stock_research.hypotheses import registry

CREDENTIALS = ("MKK_API_KEY", "MKK_API_SECRET")


def load_credentials(env_file: Path) -> None:
    """Put the two credentials into ``os.environ``. Values are never shown."""

    found = {}
    for line in Path(env_file).read_text(encoding="utf-8").splitlines():
        name, separator, value = line.strip().partition("=")
        if separator and name.strip() in CREDENTIALS:
            found[name.strip()] = value.strip().strip('"').strip("'")
    missing = [name for name in CREDENTIALS if not found.get(name)]
    if missing:
        raise SystemExit(f"{env_file}: missing {', '.join(missing)}")
    os.environ.update(found)


def other_ingestion_processes() -> list:
    """Command lines of other processes running the KAP ingester."""

    script = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "
              "'*stock_research.data.kap*' -or $_.CommandLine -like '*ingest-kap*' -or "
              "$_.CommandLine -like '*v3_validation.run_census*' -or "
              "$_.CommandLine -like '*run_anchor_ingest*' } | "
              "ForEach-Object { \"$($_.ProcessId)`t$($_.Name)`t$($_.CommandLine)\" }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                         capture_output=True, text=True).stdout
    mine = {os.getpid(), os.getppid()}
    rows = []
    for line in out.splitlines():
        pid, _, rest = line.partition("\t")
        name, _, command = rest.partition("\t")
        if pid.strip().isdigit() and int(pid) not in mine and name.lower().startswith("python"):
            rows.append(f"{pid} {command[-90:]}")
    return rows


def preconditions(db: Path) -> dict:
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        protocols = {r[0]: r[1] for r in con.execute(
            "SELECT protocol_hash, version FROM sr_protocols")}
        results = con.execute("SELECT COUNT(*) FROM sr_results").fetchone()[0]
        frame = dict(con.execute(
            "SELECT status, COUNT(*) FROM sr_kap_frame WHERE frame_version = ? GROUP BY 1",
            (KAP_ANCHOR_FRAME_VERSION,)).fetchall())
        refusals = con.execute("SELECT COUNT(*) FROM sr_kap_detail_failures").fetchone()[0]
        placeholders = con.execute(
            "SELECT COUNT(*) FROM sr_raw_kap_detail WHERE published_raw IS NULL").fetchone()[0]
        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
        # A process this check does not recognise still leaves a trace: a
        # detail stored in the last few minutes means something is fetching.
        latest = con.execute("SELECT MAX(retrieved_at) FROM sr_raw_kap_detail").fetchone()[0]
    finally:
        con.close()
    lineage = [entry["protocol_hash"] for entry in registry.LINEAGE]
    from datetime import datetime, timezone
    idle = (datetime.now(timezone.utc) - datetime.fromisoformat(latest)).total_seconds()
    checks = {
        "database_is_the_research_database": all(h in protocols for h in lineage),
        "no_stored_results": results == 0,
        "v3_not_registered": registry.protocol_hash() not in protocols,
        "anchor_frame_complete": frame == {kap.STATUS_COMPLETE: KAP_ANCHOR_FRAME_BLOCKS},
        "no_refused_or_placeholder_details": refusals == 0 and placeholders == 0,
        "integrity_check_ok": integrity == "ok",
        "no_other_ingestion_process": not other_ingestion_processes(),
        "no_detail_stored_in_the_last_five_minutes": idle >= 300,
    }
    return {"checks": checks, "anchor_frame": frame, "other_processes": other_ingestion_processes(),
            "passed": all(checks.values())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", required=True)
    parser.add_argument("--env")
    parser.add_argument("--max-hours", type=float, default=48.0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="skip the anchor-frame precondition once the census has begun")
    args = parser.parse_args(argv)
    db = Path(args.db).resolve()
    if not db.exists():
        raise SystemExit(f"{db} does not exist")
    found = preconditions(db)
    if args.resume:
        found["passed"] = all(v for k, v in found["checks"].items()
                              if k != "anchor_frame_complete")
    print(json.dumps({"database": str(db), **found}, indent=1), flush=True)
    if args.check:
        return 0 if found["passed"] else 3
    if not found["passed"]:
        return 3
    if not args.env:
        raise SystemExit("--env is required to start the census")
    load_credentials(Path(args.env))
    log = REPOSITORY_ROOT / "outputs" / "stock_research" / "v3" / "census.log"   # not committed
    log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(log, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])
    store.init_db(str(db))                      # additive: CREATE TABLE IF NOT EXISTS only
    result = kap.run(str(db), max_hours=args.max_hours)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
