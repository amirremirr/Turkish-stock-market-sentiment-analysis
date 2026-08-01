"""Compatibility entry point for selection-versus-framing inference.

The former matcher could reuse a pro-government headline in several pairs and
described lexical candidates as known same stories.  The maintained matcher is
deterministic, one-to-one, and labels those candidates as an unverified
sensitivity unless explicit repeated canonical event IDs exist.
"""

from __future__ import annotations

import sys
from typing import Sequence

from analysis.polarization.inference import (
    lexical_date_pairs,
    main as inference_main,
    significant_tokens,
)
from config import DB_PATH


MIN_SHARED = 2
MIN_LEN = 5
WINDOW = 1


def sig_tokens(title):
    """Retain the old token-helper name for notebooks."""

    return set(significant_tokens(title, minimum_length=MIN_LEN))


def match_pairs(frame):
    """Compatibility helper returning audited one-to-one fallback pairs."""

    return lexical_date_pairs(
        frame,
        minimum_shared_tokens=MIN_SHARED,
        window_days=WINDOW,
        minimum_token_length=MIN_LEN,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--db" not in args:
        args[0:0] = ["--db", str(DB_PATH)]
    return inference_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
