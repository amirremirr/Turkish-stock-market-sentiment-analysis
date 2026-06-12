# KAP API research spike (Migration Phase 3) — 2026-06-13

## Access route

The official KAP Veri Yayın Servisi requires a Borsa İstanbul data contract —
closed to individuals. The open route is the **MKK API Portal**
(https://apiportal.mkk.com.tr): free self-service signup, automatic app
approval. Our app `bist-sentiment` is registered on the free plan.

- **Plan:** Ücretsiz — throttled to **6 calls/minute** (ample: we poll a few
  times per scheduled run)
- **Gateway:** `https://apigwdev.mkk.com.tr/api/vyk` (dev/test environment;
  production cutover via kapdestek@mkk.com.tr later)
- **Auth:** HTTP **Basic** (app credentials from the portal's Uygulamalarım page)
- **Spec:** OpenAPI 3.0.3, public, saved as `kap_openapi.json`
  (`https://apigwdev.mkk.com.tr/api/vyk?openapi`)

## Endpoints (11)

| Endpoint | Use |
|---|---|
| `GET /lastDisclosureIndex` | Latest published disclosure ID — cheap poll |
| `GET /disclosures?disclosureIndex=N` | All disclosures AFTER index N (+ optional `disclosureTypes`, `disclosureClass`, `companyId[]`) |
| `GET /disclosureDetail/{index}?fileType=` | Full disclosure content |
| `GET /members` | Listed-company master — feeds Phase 6 entity linking |
| `GET /memberDetail/{id}`, `/memberSecurities` | Company detail, ISIN securities |
| `GET /funds`, `/fundDetail/{id}` | Funds |
| `GET /downloadAttachment/{id}` | Attachments |
| `GET /blockedDisclosures`, `/caEventStatus` | Housekeeping / corporate actions |

## Ingestion design (ingest/kap.py)

Incremental cursor pattern, built into the API:

1. Store `last_seen_index` in DB (new `kap_state` row or reuse experiments)
2. Each run: `GET /disclosures?disclosureIndex=<last_seen>` → new disclosures
3. Insert as **Tier A events** (no headline_id): `credibility=1.0`,
   `event_type` from disclosure class/type code, `signal_date` from publish
   timestamp via trading_calendar
4. Update cursor; respect 6/min throttle (sleep ~12s between calls)

First run seeds the cursor from `lastDisclosureIndex` minus a small window —
the dev environment's history depth is unknown; verify on first authenticated
call.

## Findings from authenticated probing (2026-06-13)

- **Dev gateway = historical SAMPLE dataset** ending late December 2023
  (lastDisclosureIndex 1231017 → 2023-12-31 disclosures). Good for
  integration work; useless for live signal. **Production access** is the
  next step: email kapdestek@mkk.com.tr referencing the portal app.
- Server-side filtering works: `disclosures?disclosureTypes=ODA` (comma-join
  for multiple). Types seen: ODA (material events), FR, FON, DG classes.
- Detail payload verified: `time` is `dd.MM.yyyy HH:mm:ss`, `subject.tr`,
  `summary.tr`, `senderTitle`, `relatedStocks[].code` (often empty in sample).
- Auth = plain HTTP Basic with portal app key/secret. Throttle 6/min enforced.

## Implementation status

`kap_ingest.py` built and dry-run validated against the sample data
(`python main.py kap-ingest --dry-run`). Guarded by `KAP_ENABLED=False` so
sample-era events can never pollute the research store. Cursor in
`kv_state['kap_cursor']`; events deduped on `external_id='kap:<index>'`;
relatedStocks land in `event_entities` as tickers.
