# Entity mapping: from text to ticker

Implemented in `stock_research/entities.py` (news) and
`stock_research/events.py` (KAP disclosures).

A wrong ticker puts another company's return into an event study, and no
statistical method repairs that. The mapping therefore prefers a missed link
to a wrong one.

## KAP disclosures: no text matching

A disclosure carries the filer's identifier and exchange codes. The issuer is
taken from those fields, never from the text.

Two cases need care, both found by reading real forms:

- **The filer is not always the company the event is about.** On a share
  transaction notification a parent or holder files, and the traded company
  is in the "related companies" field. The ticker comes from that field. If
  the form names more than one other company, no ticker is assigned.
- **An issuer can have several listed lines** (share classes, a warrant
  issuer code). The line used is the one with the highest median turnover over
  the 60 sessions before day 0. The rule looks only backwards and needs no
  hand-made table.

Codes are those the provider serves today. A company that has since changed
its code or been delisted has no price series and drops out; that is a
survivorship limitation, not a mapping error.

## News: ranked evidence

| Rank | Evidence | Example | Confidence |
|---|---|---|---|
| 1 | Issuer identifier from the source | KAP sender | not a text match |
| 2 | Ticker token in capitals in mixed-case text | `THYAO`, `(ASELS)` | 0.97 |
| 3 | Unambiguous alias | `turkcell`, `tupras` | 0.93 |
| 4 | Ambiguous alias beside a corporate cue | `koc holding`, `sok market` | 0.85 |
| 5 | Ambiguous alias, no cue | `Koç burcu`, `Şok gelişme` | 0.40: **not a link**, queued for review |

A link is confirmed at 0.80 or above. Anything lower is in a review queue and
never in a sample.

Details that matter:

- **Capitals carry information only in mixed-case text.** In an all-capitals
  headline a capitalised token is not treated as a ticker.
- **A roster code can be an ordinary word** (`ALTIN`, `METRO`). Outside the
  curated list a code must be in brackets or beside a word about shares
  (`hisse`, `hedef fiyat`, …).
- **Ambiguous aliases are listed by hand.** A name belongs there if a Turkish
  reader could meet it in a headline that is not about the company: family
  names (`Koç`, `Sabancı`, `Ülker`), ordinary words (`şok`, `mavi`, `logo`,
  `garanti`), other things with the same name (`Pegasus`, `Astor`).
- **Related institutions are not the issuer.** Sabancı University and Enka
  sports club are not Sabancı Holding and Enka İnşaat.

## Mention type

A link says the text names the company. Whether the text is a material
announcement about it is a separate judgement.

| Type | Meaning | Admissible as an event |
|---|---|---|
| `material` | One issuer, with a material-event cue (results, dividend, contract, capital increase, acquisition, penalty, …) | yes |
| `casual` | One issuer, no such cue | no |
| `multi_issuer` | More than one issuer linked | no (each link is still recorded) |
| `sector` | Sector vocabulary and no material cue | no |

Macro news links to no issuer and never enters a stock-level sample.

Duplicate stories across outlets are meant to be collapsed to one story before
counting (one `story_id`); the coverage measure in H4 counts stories, not
copies.

## Evaluation

`python -m stock_research.cli evaluate-linker` scores the linker on
`tests/stock_research/entity_eval.csv`: 77 hand-written headlines, 60
gold (headline, ticker) pairs, 22 with no issuer at all. The negatives are
the hard ones: ordinary-word uses of company names, family and institution
names, place names, sector and macro headlines.

| Metric | Value |
|---|---|
| Precision | 1.00 |
| Recall | 0.983 |
| F1 | 0.992 |
| False ticker assignments | 0 |
| Unresolved (neither confirmed nor queued) | 0 |
| Queued for review | 1 |

**Read this table for what it is.** The examples were written by the same
author as the rules, so it is a development set. It shows the rules do what
they were written to do. It does not estimate accuracy on real headlines; that
needs a sample of real headlines labelled by someone else.

On the real corpus (3,727 scored headlines outside the sealed window) the
linker confirms an issuer in 84 and queues 20 for review. Of the 84, 7 are
single-issuer material mentions. No accuracy figure is claimed for those.

## Known gaps

- The curated alias list covers 55 large and mid-cap issuers. Smaller issuers
  are reachable only through a ticker token.
- No named-entity model and no contextual disambiguation beyond the cue
  window. Both were considered and deferred: with 7 usable events there is
  nothing for them to improve.
- No manual-review tool, only the queue.
- No historical alias or ticker-change dictionary.
