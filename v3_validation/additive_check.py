"""Is the research database still everything the pre-census backup was?

    python -m v3_validation.additive_check <database> <backup>

Read-only on both. For every table of the backup: each of its rows must still
be in the database, unchanged. Rows may have been added; none may have been
altered or removed. The frame bookkeeping table and the key-value table are
mutable by design, so for them only the earlier frame's rows and the earlier
keys are compared. New tables are listed.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

from stock_research.config import KAP_FRAME_VERSION


def _rows(con, table: str, where: str = "") -> dict:
    keys = [r[1] for r in con.execute(f"PRAGMA table_info({table})") if r[5]]
    columns = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    key_of = (lambda row: tuple(row[columns.index(k)] for k in keys)) if keys else (lambda row: row)
    return {key_of(row): hashlib.sha256(repr(row).encode("utf-8")).hexdigest()
            for row in con.execute(f"SELECT * FROM {table} {where}")}


def check(database: Path, backup: Path) -> dict:
    live = sqlite3.connect(f"file:{Path(database).as_posix()}?mode=ro", uri=True)
    old = sqlite3.connect(f"file:{Path(backup).as_posix()}?mode=ro", uri=True)
    try:
        tables_old = [r[0] for r in old.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1")]
        tables_live = [r[0] for r in live.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1")]
        out = {"tables": {}, "new_tables": sorted(set(tables_live) - set(tables_old)),
               "tables_removed": sorted(set(tables_old) - set(tables_live))}
        for table in tables_old:
            where = ""
            if table == "sr_kap_frame":
                where = f"WHERE frame_version != '{KAP_FRAME_VERSION}'"
            before, after = _rows(old, table, where), _rows(live, table, where)
            if table == "sr_meta":
                after = {k: v for k, v in after.items() if k in before}
            out["tables"][table] = {
                "rows_in_backup": len(before), "rows_now": len(_rows(live, table)),
                "backup_rows_missing_now": sum(1 for k in before if k not in after),
                "backup_rows_changed": sum(1 for k, v in before.items()
                                           if k in after and after[k] != v)}
        triggers = lambda con: {r[0] for r in con.execute(                      # noqa: E731
            "SELECT name FROM sqlite_master WHERE type='trigger'")}
        out["triggers_removed"] = sorted(triggers(old) - triggers(live))
        out["integrity_check"] = live.execute("PRAGMA integrity_check").fetchone()[0]
        out["additive"] = (not out["tables_removed"] and not out["triggers_removed"]
                           and out["integrity_check"] == "ok"
                           and all(t["backup_rows_missing_now"] == 0 and t["backup_rows_changed"] == 0
                                   for t in out["tables"].values()))
        return out
    finally:
        live.close()
        old.close()


if __name__ == "__main__":
    result = check(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(result, indent=1))
    raise SystemExit(0 if result["additive"] else 3)
