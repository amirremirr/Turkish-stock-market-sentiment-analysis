"""Constants for the stock-level study. Anything that affects a result is here
and is hashed into the registered protocol (see hypotheses/registry.py)."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

#: Separate database file. Holds licensed raw data, so it is gitignored and
#: never committed.
STOCK_DB_PATH = REPOSITORY_ROOT / "stock_research.db"

SCHEMA_VERSION = "stock-research-schema-v2"

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
#:
#: v3 is a census of the same range: every material-event disclosure of a
#: listed company between the two indices, read page by page. The readiness
#: audit measured about 15,100 of them (0.108 per disclosure number), which is
#: about two days of calls, not the infeasible job the v2 note assumed; and it
#: showed that 80 anchors cannot detect an effect of plausible size. Every
#: listing row and detail stored under v2 is a member of the census and is
#: reused as stored.
KAP_FRAME_VERSION = "kap-census-2023-v3"
KAP_FRAME_FIRST_INDEX = 1_091_700
KAP_FRAME_LAST_INDEX = 1_231_017
KAP_LISTING_PAGE = 50               # disclosures per listing call
#: Earlier frames whose raw rows stay in the database. Their listing rows are
#: census members by construction (same range, same listing filter).
KAP_EARLIER_FRAMES = ("kap-anchor-sample-v2",)
KAP_ANCHOR_FRAME_VERSION = "kap-anchor-sample-v2"
KAP_ANCHOR_FRAME_BLOCKS = 80
KAP_ANCHOR_FRAME_STRIDE = 37
KAP_LISTING_TYPES = ("ODA", "CA")   # material events and structured corporate-action forms
KAP_THROTTLE_SECONDS = 11           # free plan: 6 calls/minute
KAP_TARGET_CLASSES = ("ODA",)
KAP_LISTED_MEMBER_MARK = "IGS"      # "İşlem Gören Şirket": listed company
#: How many times a disclosure detail the gateway refuses is asked for again.
KAP_MAX_DETAIL_ATTEMPTS = 3
#: A census with a hole is not a census. The run needs every target
#: disclosure retrieved: the share that may be unavailable is zero. A
#: disclosure the gateway refuses is asked for again on
#: ``KAP_MAX_DETAIL_ATTEMPTS`` separate days; if it is still missing the run
#: refuses, and whether to proceed without it is a decision to be reviewed
#: and registered as a new protocol, not a tolerance applied automatically
#: (condition 6 of the approval of 2026-10-11). v2 allowed 1% after a single
#: attempt; with categories of 30 events that could have been most of a
#: category.
KAP_MAX_UNAVAILABLE_SHARE = 0.0
KAP_MIN_ATTEMPT_DAYS = 3

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

# -- H7, protocol v3 -----------------------------------------------------------
#: Draws and seed of the wild cluster bootstrap. With 9,999 draws the smallest
#: attainable p-value is 0.0001, far below the 0.0056 a lone test must reach.
WILD_BOOTSTRAP_DRAWS = 9999
WILD_BOOTSTRAP_SEED = 20261011
#: A category is tested only if it has this many events, on this many
#: distinct day-0 sessions, from this many distinct issuers, after an issuer's
#: overlapping events have been thinned to one. v2 had no issuer rule.
H7_MIN_EVENTS = 30
H7_MIN_DATES = 20
H7_MIN_ISSUERS = 20
#: The reference group must be larger: every contrast is taken against it.
H7_REFERENCE_MIN_EVENTS = 60
H7_REFERENCE_MIN_DATES = 30
H7_REFERENCE_MIN_ISSUERS = 30

# -- Prices, protocol v3 -------------------------------------------------------
#: A return enters the study only if two providers agree on it to within this
#: many return points. The readiness audit found the primary provider's close
#: was not the closing price for most stocks on 16 dates; the cleaning rules
#: caught 26 of about 1,150 wrong returns.
RECONCILE_TOLERANCE = 0.01

# -- Costs (scenarios, not measured fees) --------------------------------------
ROUND_TRIP_COST_SCENARIOS = (0.003, 0.004, 0.005)
