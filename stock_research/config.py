"""Constants for the stock-level study. Anything that affects a result is here
and is hashed into the registered protocol (see hypotheses/registry.py)."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

#: Separate database file. Holds licensed raw data, so it is gitignored and
#: never committed.
STOCK_DB_PATH = REPOSITORY_ROOT / "stock_research.db"

SCHEMA_VERSION = "stock-research-schema-v1"

# -- Sealed index test ---------------------------------------------------------
#: The index study's untouched window opens here. Until its single result is
#: recorded, no stock-level outcome on or after this session may be computed:
#: stock returns around news are close enough to the sealed index target that
#: reading them would be a look at it.
SEALED_INDEX_BOUNDARY = "2026-08-10"

# -- KAP sample ----------------------------------------------------------------
#: The dev gateway serves a historical sample; these bounds were measured on
#: 2026-10-11 (docs/stock_research/DATA_AVAILABILITY.md). One detail call per
#: disclosure at 6 calls/minute makes a census infeasible, so the frame is a
#: systematic sample fixed before any return was read: evenly spaced anchor
#: indices, and at each anchor the next 50 material-event disclosures.
#:
#: v1 listed unfiltered blocks and was abandoned after 9 blocks, before any
#: price was read: two thirds of all disclosures are daily fund bulletins, so
#: a block held about four usable events. v2 filters the listing by type.
KAP_FRAME_VERSION = "kap-anchor-sample-v2"
KAP_FRAME_FIRST_INDEX = 1_091_700
KAP_FRAME_LAST_INDEX = 1_231_017
KAP_FRAME_BLOCKS = 80
KAP_FRAME_STRIDE = 37               # coprime with 80: processing order spreads over the year
KAP_LISTING_TYPES = ("ODA", "CA")   # material events and structured corporate-action forms
KAP_THROTTLE_SECONDS = 11           # free plan: 6 calls/minute
KAP_TARGET_CLASSES = ("ODA",)
KAP_LISTED_MEMBER_MARK = "IGS"      # "İşlem Gören Şirket": listed company

# -- Event study ---------------------------------------------------------------
EVENT_WINDOWS = (
    (-5, -1), (0, 0), (0, 1), (2, 5), (0, 5), (1, 10), (1, 20),
)
ESTIMATION_WINDOW = (-130, -11)     # sessions relative to day 0, strictly pre-event
ESTIMATION_MIN_OBSERVATIONS = 60
BENCHMARK_TICKER = "XU100.IS"

# -- Inference -----------------------------------------------------------------
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261011
ALPHA = 0.05
#: Below these an estimate is reported as data_insufficient, never tested.
MIN_EVENTS_PER_GROUP = 30
MIN_EVENT_DATES_PER_GROUP = 15

# -- Costs (scenarios, not measured fees) --------------------------------------
ROUND_TRIP_COST_SCENARIOS = (0.003, 0.004, 0.005)
