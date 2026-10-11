# Data dictionary

All tables are in `stock_research.db`. Raw tables (prefix `sr_raw_`),
`sr_protocols`, `sr_manifests` and `sr_results` are append-only: an `UPDATE`
or `DELETE` is refused by trigger.

## Raw tables

### `sr_raw_kap_listing`
One row per disclosure returned by a listing call.

| Column | Meaning |
|---|---|
| `disclosure_index` | KAP's disclosure number (primary key) |
| `frame_version`, `block_seq` | Which sample frame and anchor it belongs to |
| `disclosure_type`, `disclosure_class` | As returned: `ODA`, `CA`, … |
| `company_id`, `title` | Filer's member id and name |
| `payload_json` | The listing row verbatim |
| `source`, `retrieved_at` | Provenance |

The listing endpoint returns no timestamp.

### `sr_raw_kap_detail`
One row per fetched disclosure.

| Column | Meaning |
|---|---|
| `sender_id`, `sender_title`, `sender_codes` | The filer and its exchange codes (JSON list) |
| `related_stocks` | The form's "related companies" field (JSON list of `{"code": …}`) |
| `subject_tr`, `subject_en`, `summary_tr` | KAP's form name and the filer's one-line summary |
| `published_raw` | `dd.MM.yyyy HH:mm:ss`, Istanbul local time, as returned |
| `event_type_code` | KAP's structured event code, where present |
| `body_text` | Turkish text extracted from the HTML body |
| `metadata_json` | Every field except the HTML body |
| `payload_sha256` | Hash of the original payload |

The HTML body is not stored (about 100 KB each); its text and hash are.

### `sr_raw_members`
The provider's roster at `retrieved_at`: `member_id`, `title`, `stock_codes`,
`member_type`. **Not point-in-time.** Companies delisted before the snapshot
are absent and codes are current.

### `sr_raw_price_bars`
Daily bars as returned, per snapshot: `open`, `high`, `low`, `close`,
`adj_close`, `volume`, `dividend`, `split_ratio`, `source`, `retrieved_at`.
Primary key `(snapshot_id, ticker, date)`.

`close` is split-adjusted by the provider. `adj_close` is also
dividend-adjusted.

### `sr_price_availability`
One row per requested symbol and snapshot: `status` is `ok`, `no_data` or
`error`. A symbol the provider could not serve is recorded, not skipped.

## Bookkeeping

- `sr_kap_frame`: progress of each anchor (`pending`, `listed`, `complete`).
  Mutable.
- `sr_protocols`: registered protocol documents by hash.
- `sr_manifests`: what each stored run was computed from.
- `sr_results`: one row per manifest and hypothesis, with the full result JSON.

## The analysis panel (in memory)

Built by `data/prices.py::build_panel`. One frame per ticker, indexed by every
session of the benchmark; a session the ticker did not trade is a row of NaN.

| Column | Definition | Missing when |
|---|---|---|
| `ret` | `adj_close_t / adj_close_{t-1} − 1` | either side missing; unexplained jump; move beyond the price limit; structurally bad bar |
| `raw_ret` | `close_t / close_{t-1} − 1` | as above, and on corporate-action dates and merged sessions |
| `gap` | `log(open_t / close_{t-1})` | as `raw_ret` |
| `intraday` | `log(close_t / open_t)` | as `raw_ret` |
| `turnover` | `close × volume` | no bar |
| `corporate_action` | dividend or split recorded that day | never |

Panel-level fields: `calendar_gaps` (dates most stocks traded but the
benchmark lacks) and `merged_sessions` (the sessions that absorbed them).

## The KAP event table (in memory)

Built by `events.py::build_kap_events`, one row per sampled disclosure.

| Column | Meaning |
|---|---|
| `event_id` | `kap:<disclosure_index>` |
| `ticker` | Provider symbol of the issuer the event is about; null if unresolved |
| `sender_codes`, `related_codes`, `names_other_issuer` | Who filed and whom the form names |
| `published_local`, `published_utc` | Publication time in `Europe/Istanbul` and UTC |
| `day0` | First session able to act on it |
| `bucket` | `pre_open`, `during_session`, `post_close`, `weekend_or_holiday`, `unknown` |
| `timing_ambiguous` | Afternoon of a possible half day, or outside the calendar |
| `category`, `category_rule` | Taxonomy category and the rule that assigned it |
| `is_update`, `is_correction`, `is_delayed` | The form's own flags; null if the form has none |
| `insider_*` | Share-transaction fields: `parse_status`, `kind`, `side`, `buy_nominal`, `sell_nominal`, `price_mid`, `value`, `role`, `exclusion` |

## News events (input to H1–H6, H9)

One row per deduplicated story and issuer:

| Column | Meaning |
|---|---|
| `event_id`, `story_id` | Story identifier; syndicated copies share one |
| `ticker`, `subject` | Linked issuer |
| `day0`, `bucket`, `timing_ambiguous` | As for KAP events |
| `sentiment` | Stored score in [−1, +1] |
| `mention_type` | `material`, `casual`, `sector`, `multi_issuer` |
| `n_issuers`, `link_confirmed` | Linker output |
| `first_published_utc`, `language` | For the cross-language test |

No real table of this kind exists yet; see DATA_AVAILABILITY.md.

## Result JSON

Each hypothesis returns: `status`, `data_origin`, `frame_complete`,
`claim_allowed`, `sufficiency` (met, reasons, counts), `primary` (estimate,
se, ci, p, p_holm, n, clusters), `exploratory`, `sensitivity`, `execution`,
`limitations`, `tables`.
