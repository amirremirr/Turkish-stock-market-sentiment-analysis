"""Resume the v2 anchor ingestion from this worktree against an explicit database.

    python run_anchor_ingest.py <path-to-stock_research.db> <path-to-.env> [max-hours]

The two gateway credentials are read from the given .env into this process's
environment only; they are not printed, logged or written anywhere.
"""

import json
import logging
import os
import sys
from pathlib import Path

from stock_research.config import REPOSITORY_ROOT
from stock_research.data import kap


def main() -> int:
    db, env = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    hours = float(sys.argv[3]) if len(sys.argv) > 3 else 24.0
    if not db.exists():
        raise SystemExit(f"{db} does not exist")
    for line in env.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.strip().partition("=")
        if separator and name.strip() in ("MKK_API_KEY", "MKK_API_SECRET"):
            os.environ[name.strip()] = value.strip().strip('"').strip("'")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(REPOSITORY_ROOT / "stock_research_ingest.log",
                                      encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    print(json.dumps(kap.run(str(db), max_hours=hours), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
