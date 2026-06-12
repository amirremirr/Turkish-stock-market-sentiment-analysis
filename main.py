"""
BIST 100 Sentiment Pipeline - CLI entry point

Usage
-----
  python main.py run                      # full pipeline (recommended first run)
  python main.py scrape                   # fetch latest headlines only
  python main.py score                    # run sentiment model on unscored headlines
  python main.py aggregate                # recompute daily sentiment aggregates
  python main.py prices                   # download BIST 100 price history
  python main.py plot                     # generate visualisation
  python main.py status                   # show DB statistics
  python main.py clean                    # remove off-topic headlines from DB
  python main.py clean --dry-run          # preview how many would be removed
  python main.py export-labels            # export stratified CSV for model validation
  python main.py export-labels --n 300    # export 300 headlines (~100 per label)

Global flags
------------
  --db PATH        SQLite file path (default: finance_sentiment.db)
  --days N         Lookback window in days  (default: 90)
  --output PATH    Plot output file          (default: sentiment_vs_bist100.png)
  --no-show        Don't open the interactive plot window
  --log-level      DEBUG | INFO | WARNING    (default: INFO)
"""

import argparse
import logging
import sys
from pathlib import Path

import database as db
import pipeline as p
from config import DB_PATH, DEFAULT_LOOKBACK_DAYS, PLOT_OUTPUT


# -- Logging setup -------------------------------------------------------------

def _setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        level=numeric,
    )
    # Keep yfinance and transformers quieter at INFO
    if numeric >= logging.INFO:
        logging.getLogger("yfinance").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("filelock").setLevel(logging.WARNING)


# -- Subcommand handlers -------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    print("\n>>  Running full sentiment pipeline ...\n")
    p.run_all(
        lookback_days=args.days,
        db_path=args.db,
        output_path=args.output,
        show_plot=not args.no_show,
    )
    print("\n[OK]  Pipeline complete.\n")


def cmd_scrape(args: argparse.Namespace) -> None:
    db.init_db(args.db)
    n = p.scrape_step(lookback_days=args.days, db_path=args.db)
    print(f"  {n} new headlines scraped and stored.")


def cmd_score(args: argparse.Namespace) -> None:
    db.init_db(args.db)
    n = p.score_step(db_path=args.db)
    print(f"  {n} headlines scored.")


def cmd_aggregate(args: argparse.Namespace) -> None:
    db.init_db(args.db)
    n = p.aggregate_step(db_path=args.db)
    print(f"  {n} days of daily sentiment computed.")


def cmd_recategorize(args: argparse.Namespace) -> None:
    """Re-classify every headline — keyword rules by default, LLM with --llm."""
    db.init_db(args.db)
    if getattr(args, "llm", False):
        result = p.recategorize_llm_step(db_path=args.db)
        print(f"  {result['recategorized']} categories changed by the LLM.")
        if result["low_relevance"]:
            print(f"  {len(result['low_relevance'])} headlines graded below the "
                  f"aggregation threshold (kept in DB, weight ~0):")
            for grade, t in result["low_relevance"]:
                # ascii-fold for Windows consoles that aren't UTF-8
                print(f"    [{grade:.1f}] {t[:85]}".encode("ascii", "replace").decode())
        return
    n = p.recategorize_step(db_path=args.db, force=True)
    print(f"  {n} headlines recategorized.")
    print("  Re-running aggregate to refresh daily_sentiment ...")
    days = p.aggregate_step(db_path=args.db)
    print(f"  {days} days of sentiment recomputed.")


def cmd_relabel(args: argparse.Namespace) -> None:
    """
    Recompute sentiment_label for all scored headlines from stored
    probabilities using the CURRENT config thresholds, then re-aggregate.
    Run this after changing SENTIMENT_*_THRESHOLD in config.py.
    """
    from config import SENTIMENT_POSITIVE_THRESHOLD, SENTIMENT_NEGATIVE_THRESHOLD

    db.init_db(args.db)
    n = db.relabel_from_probs(
        SENTIMENT_POSITIVE_THRESHOLD, SENTIMENT_NEGATIVE_THRESHOLD, db_path=args.db
    )
    print(f"  {n} headline label(s) updated to current thresholds "
          f"(+{SENTIMENT_POSITIVE_THRESHOLD} / {SENTIMENT_NEGATIVE_THRESHOLD}).")
    if n:
        print("  Re-running aggregate to refresh daily_sentiment ...")
        days = p.aggregate_step(db_path=args.db)
        print(f"  {days} days of sentiment recomputed.")


def cmd_prices(args: argparse.Namespace) -> None:
    db.init_db(args.db)
    n = p.prices_step(lookback_days=args.days, db_path=args.db)
    print(f"  {n} trading-day price rows stored.")


def cmd_plot(args: argparse.Namespace) -> None:
    db.init_db(args.db)
    path = p.plot_step(
        days=args.days,
        output_path=args.output,
        db_path=args.db,
        show=not args.no_show,
    )
    if path:
        print(f"  Saved -> {path}")
    else:
        print("  [!!]  Not enough overlapping data to plot yet.")
        print("     Run  'python main.py run'  to populate the database first.")


def cmd_fx_rates(args: argparse.Namespace) -> None:
    db.init_db(args.db)
    n = p.fx_rates_step(lookback_days=args.days, db_path=args.db)
    if n:
        print(f"  {n} USD/TRY FX rate days stored.")
    else:
        print("  No FX data retrieved (check ALPHA_VANTAGE_KEY or rate limit).")


def cmd_clean(args: argparse.Namespace) -> None:
    db.init_db(args.db)
    if args.dry_run:
        n = p.clean_step(db_path=args.db, dry_run=True)
        print(f"  [dry-run] {n} headlines would be removed by the current filter.")
        print("  Run  'python main.py clean'  (without --dry-run) to delete them.")
    else:
        n = p.clean_step(db_path=args.db, dry_run=False)
        if n == 0:
            print("  No off-topic headlines found - DB is already clean.")
        else:
            print(f"  {n} off-topic headlines removed. Daily sentiment re-aggregated.")


def cmd_export_labels(args: argparse.Namespace) -> None:
    """
    Export a stratified random sample of scored headlines as a CSV for
    manual model-validation labelling.

    The CSV has columns:
        id            - headline DB id (for traceability)
        source        - RSS source key
        published_at  - publication date
        title         - original headline text (Turkish)
        model_label   - label assigned by the sentiment model
        model_score   - continuous score [−1, +1]
        human_label   - BLANK — annotator fills this in

    Workflow:
        1. python main.py export-labels --n 300
        2. Open labels_to_validate.csv in Excel / Google Sheets
        3. Fill the human_label column: positive / neutral / negative
        4. Compute accuracy: (model_label == human_label).mean()
        5. Break down by category to find where the model fails
    """
    import csv
    import pandas as pd

    db.init_db(args.db)

    with db._conn(args.db) as con:
        df = pd.read_sql_query(
            """SELECT id, source, published_at, title,
                      sentiment_label AS model_label,
                      ROUND(sentiment_score, 4) AS model_score,
                      ROUND(COALESCE(relevance, 1.0), 2) AS model_relevance,
                      category
               FROM headlines
               WHERE sentiment_label IS NOT NULL
               ORDER BY RANDOM()""",
            con,
        )

    if df.empty:
        print("  No scored headlines. Run 'python main.py score' first.")
        return

    # Exclude already-labeled headlines (pass the previous labels CSV) so a new
    # export only contains fresh work.
    if getattr(args, "exclude", None):
        try:
            done_ids = set(pd.read_csv(args.exclude)["id"].astype(int))
            before_excl = len(df)
            df = df[~df["id"].astype(int).isin(done_ids)]
            print(f"  [exclude] Dropped {before_excl - len(df)} already-labeled headlines.")
        except Exception as exc:
            print(f"  [!] Could not read exclude file: {exc}")

    # Deduplicate on normalized title[:80] so the label set has no near-duplicates.
    # This is important for the 300-500 label path: if the same story appears
    # multiple times (from different sources), we only keep the first occurrence.
    from scraper import _normalise as _sc_norm
    df["_title_hash"] = df["title"].apply(lambda t: _sc_norm(str(t))[:80])
    before = len(df)
    df = df.drop_duplicates(subset="_title_hash").drop(columns=["_title_hash"])
    after = len(df)
    if before > after:
        print(f"  [dedup] Dropped {before - after} near-duplicate titles "
              f"before sampling ({after} unique remain).")

    # Stratified sample: equal representation of pos / neu / neg
    n_per_label = max(1, args.n // 3)
    frames = []
    for label in ["positive", "neutral", "negative"]:
        subset = df[df["model_label"] == label]
        n_take = min(n_per_label, len(subset))
        if n_take:
            frames.append(subset.head(n_take))
        else:
            print(f"  [!]  No '{label}' headlines available to sample.")

    if not frames:
        print("  No scored headlines in the expected label classes.")
        return

    sample = pd.concat(frames, ignore_index=True)
    # Shuffle so the annotator sees labels interleaved, not in groups
    sample = sample.sample(frac=1, random_state=42).reset_index(drop=True)
    sample["human_label"] = ""
    # Validate the LLM's relevance grade alongside sentiment: mark y/n whether
    # the headline belongs in a Turkish-market sentiment index at all.
    sample["human_relevant"] = ""

    # Reorder columns to a natural reading order
    sample = sample[["id", "source", "published_at", "category",
                      "title", "model_label", "model_score", "model_relevance",
                      "human_label", "human_relevant"]]

    # Use --output if explicitly set to something other than the plot default,
    # otherwise default to a dedicated filename that won't collide with the plot.
    from config import PLOT_OUTPUT
    out = args.output if args.output != PLOT_OUTPUT else "labels_to_validate.csv"
    # utf-8-sig: BOM prefix so Excel opens Turkish chars correctly
    sample.to_csv(out, index=False, encoding="utf-8-sig")

    breakdown = dict(sample["model_label"].value_counts())
    print(f"\n  Exported {len(sample)} headlines -> {out}")
    print(f"  Label breakdown:  {breakdown}")
    print()
    print("  Next steps:")
    print(f"    1. Open  {out}  in Excel or Google Sheets")
    print("    2. Fill the 'human_label' column: positive / neutral / negative")
    print("    3. Compute accuracy: fraction where human_label == model_label")
    print("    4. Break down errors by category to find weak spots")
    print("    5. Use errors to guide fine-tuning or rule overrides")
    print()
    print("  Validation target: >= 70% accuracy per category on 300+ headlines")


def cmd_validate_labels(args: argparse.Namespace) -> None:
    """
    Validate the sentiment model against a human-labeled CSV.

    Workflow:
      1. python main.py export-labels --n 300
      2. Open labels_to_validate.csv, fill the human_label column
      3. python main.py validate-labels labels_to_validate.csv --save
    """
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).parent / "validate_labels.py"
    cmd    = [sys.executable, str(script)]

    if getattr(args, "tracker", False):
        cmd.append("--tracker")
    else:
        if not args.labels_csv:
            print("  Usage: python main.py validate-labels <csv_file>")
            print("         python main.py validate-labels --tracker")
            return
        cmd.append(args.labels_csv)
        if getattr(args, "threshold", None) is not None:
            cmd += ["--threshold", str(args.threshold)]
        if getattr(args, "save_report", False):
            cmd.append("--save")

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    sys.exit(result.returncode)


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Generate the self-contained HTML dashboard and open it in the browser."""
    import dashboard

    db.init_db(args.db)
    path = dashboard.generate(db_path=args.db)
    print(f"  Dashboard saved -> {path}")
    if not args.no_show:
        import webbrowser
        webbrowser.open(Path(path).resolve().as_uri())


def cmd_migrate_events(args: argparse.Namespace) -> None:
    """One-time (idempotent) sync of all scored headlines into the events table."""
    import events_bridge

    db.init_db(args.db)
    n = events_bridge.sync(db_path=args.db)
    with db._conn(args.db) as con:
        total = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        tiers = con.execute(
            "SELECT source_tier, COUNT(*) FROM events GROUP BY source_tier"
        ).fetchall()
    print(f"  {n} new event(s) created; events total: {total}")
    for t in tiers:
        print(f"    tier {t[0]}: {t[1]}")


def cmd_kap_ingest(args: argparse.Namespace) -> None:
    """Ingest KAP Tier-A disclosures into the events table (or --dry-run)."""
    import kap_ingest
    from config import KAP_ENABLED

    db.init_db(args.db)
    dry = getattr(args, "dry_run", False)
    if not KAP_ENABLED and not dry:
        print("  KAP_ENABLED is False (dev gateway = historical sample data).")
        print("  Use --dry-run to validate, or enable after production access.")
        return
    result = kap_ingest.ingest(db_path=args.db, dry_run=dry)
    tag = " [DRY RUN — nothing written]" if dry else ""
    print(f"  {result['new_events']} new Tier-A event(s); cursor at {result['cursor']}{tag}")
    for ev in result["samples"][:8]:
        line = (f"    {ev['published_at']} sig={ev['signal_date']} "
                f"[{ev['event_type']}] {ev['title'][:60]} "
                f"tickers={ev['tickers']}")
        print(line.encode("ascii", "replace").decode())


def cmd_status(args: argparse.Namespace) -> None:
    db.init_db(args.db)
    stats = db.db_stats(args.db)
    col_w = 28
    sep = "-" * 46
    print(f"\n{sep}")
    print(f"  Database: {args.db}")
    print(sep)
    for key, val in stats.items():
        label = key.replace("_", " ").title()
        print(f"  {label:<{col_w}} {val}")
    print(f"{sep}\n")


# -- Argument parser -----------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    # Shared/global options - defined in a parent parser so they are inherited
    # by every subcommand and work whether placed before or after the command.
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--db",        default=DB_PATH,               help="SQLite DB path")
    shared.add_argument("--days",      default=DEFAULT_LOOKBACK_DAYS,  type=int, help="Lookback window (days)")
    shared.add_argument("--output",    default=PLOT_OUTPUT,            help="Plot output file")
    shared.add_argument("--no-show",   action="store_true",            help="Don't open plot window")
    shared.add_argument("--log-level", default="INFO",                 choices=["DEBUG", "INFO", "WARNING"])

    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="BIST 100 Turkish news sentiment pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[shared],
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    sub.add_parser("run",          parents=[shared], help="Run the full pipeline end-to-end")
    sub.add_parser("scrape",       parents=[shared], help="Scrape new headlines only")
    sub.add_parser("score",        parents=[shared], help="Run sentiment model on unscored headlines")
    sub.add_parser("aggregate",    parents=[shared], help="Recompute daily sentiment aggregates")
    recat_p = sub.add_parser("recategorize", parents=[shared],
                             help="Re-classify ALL headlines (keyword rules, or --llm) + re-aggregate")
    recat_p.add_argument("--llm", action="store_true",
                         help="Use the LLM for category + relevance (deletes irrelevant headlines)")
    sub.add_parser("relabel",      parents=[shared],
                   help="Recompute sentiment labels from stored probabilities with current thresholds")
    sub.add_parser("prices",       parents=[shared], help="Download BIST 100 price history")
    sub.add_parser("fx-rates",     parents=[shared], help="Download USD/TRY FX rates (Alpha Vantage)")
    sub.add_parser("plot",         parents=[shared], help="Generate the visualisation")
    sub.add_parser("status",       parents=[shared], help="Show database statistics")
    sub.add_parser("dashboard",    parents=[shared],
                   help="Generate the self-contained HTML dashboard (dashboard.html)")
    sub.add_parser("migrate-events", parents=[shared],
                   help="Sync scored headlines into the events table (idempotent)")
    kap_p = sub.add_parser("kap-ingest", parents=[shared],
                           help="Ingest KAP Tier-A disclosures into the events table")
    kap_p.add_argument("--dry-run", action="store_true",
                       help="Fetch and parse but write nothing (dev-gateway validation)")

    clean_p = sub.add_parser(
        "clean",
        parents=[shared],
        help="Remove off-topic headlines that fail the current relevance filter",
    )
    clean_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how many headlines would be removed without deleting anything",
    )

    export_p = sub.add_parser(
        "export-labels",
        parents=[shared],
        help="Export a stratified CSV of scored headlines for manual model validation",
    )
    export_p.add_argument(
        "--n",
        type=int,
        default=150,
        help="Total headlines to export (split equally across pos/neu/neg, default: 150)",
    )
    export_p.add_argument(
        "--exclude",
        default=None,
        help="CSV of already-labeled headlines whose ids should be skipped",
    )

    val_p = sub.add_parser(
        "validate-labels",
        parents=[shared],
        help="Validate model accuracy against a human-labeled CSV",
    )
    val_p.add_argument(
        "labels_csv", nargs="?",
        help="Path to labeled CSV (human_label column filled in)",
    )
    val_p.add_argument(
        "--threshold", type=float, default=None,
        help="Score threshold to test (default: current config value)",
    )
    val_p.add_argument(
        "--save-report", action="store_true",
        help="Save metrics snapshot to reports/validate_YYYY-MM-DD.json",
    )
    val_p.add_argument(
        "--tracker", action="store_true",
        help="Show label collection progress only (no CSV needed)",
    )

    return parser


_COMMANDS = {
    "run":              cmd_run,
    "scrape":           cmd_scrape,
    "score":            cmd_score,
    "aggregate":        cmd_aggregate,
    "recategorize":     cmd_recategorize,
    "relabel":          cmd_relabel,
    "prices":           cmd_prices,
    "fx-rates":         cmd_fx_rates,
    "plot":             cmd_plot,
    "status":           cmd_status,
    "dashboard":        cmd_dashboard,
    "migrate-events":   cmd_migrate_events,
    "kap-ingest":       cmd_kap_ingest,
    "clean":            cmd_clean,
    "export-labels":    cmd_export_labels,
    "validate-labels":  cmd_validate_labels,
}


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    _setup_logging(args.log_level)

    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as exc:
        logging.getLogger(__name__).exception("Pipeline error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
