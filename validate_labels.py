"""
validate_labels.py — Model validation against human-labeled headlines.

The export-labels command produces a CSV with a blank human_label column.
Once you fill that column in (positive / neutral / negative), run this
script to measure model quality, tune the threshold, and track progress
toward the 300-label and 500-label validation milestones.

Usage
-----
  python validate_labels.py labels_to_validate.csv         # full report
  python validate_labels.py labels_to_validate.csv --save  # also save to reports/
  python validate_labels.py labels_to_validate.csv --threshold 0.08
  python validate_labels.py --tracker                      # progress summary only
"""

import argparse
import json
import sys
import textwrap
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scraper import _normalise
from config import SENTIMENT_POSITIVE_THRESHOLD, SENTIMENT_NEGATIVE_THRESHOLD

W             = 62
VALID_LABELS  = {"positive", "neutral", "negative"}
TARGET_300    = 300
TARGET_500    = 500
LABEL_ORDER   = ["positive", "neutral", "negative"]


# -- Formatting helpers --------------------------------------------------------

def _hdr(title: str) -> None:
    print(f"\n{'=' * W}")
    print(f"  {title}")
    print(f"{'=' * W}")

def _sub(title: str) -> None:
    print(f"\n  -- {title} --")

def _row(label: str, value, note: str = "", warn: bool = False) -> None:
    tag = " [!]" if warn else ""
    print(f"  {label:<36} {str(value):<14} {note}{tag}")

def _ok(msg: str)   -> None: print(f"  [OK]  {msg}")
def _warn(msg: str) -> None: print(f"  [!!]  {msg}")
def _info(msg: str) -> None: print(f"  [ ]   {msg}")

def _progress_bar(value: float, target: int, current: int) -> str:
    pct  = min(value / target * 100, 100)
    done = int(pct / 5)
    bar  = "█" * done + "░" * (20 - done)
    need = max(0, target - current)
    note = f"  ({need} more needed)" if need > 0 else "  ✓ REACHED"
    return f"[{bar}] {pct:>5.1f}%{note}"


# -- Data loading --------------------------------------------------------------

def _load(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a labeled CSV. Returns (labeled_df, unlabeled_df).
    labeled_df is deduplicated by normalized title and filtered to valid human labels.
    """
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="utf-8")

    required = {"title", "model_score", "human_label"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}. "
                         f"Generate it with: python main.py export-labels")

    # Normalize title to dedup key
    df["_hash"] = df["title"].apply(lambda t: _normalise(str(t))[:80])
    before = len(df)
    df = df.drop_duplicates(subset="_hash").copy()
    if len(df) < before:
        _info(f"Removed {before - len(df)} near-duplicate titles before analysis.")

    df["human_label"] = df["human_label"].astype(str).str.strip().str.lower()
    labeled   = df[df["human_label"].isin(VALID_LABELS)].copy()
    unlabeled = df[~df["human_label"].isin(VALID_LABELS)].copy()

    labeled["model_score"] = pd.to_numeric(labeled["model_score"], errors="coerce")
    labeled = labeled.dropna(subset=["model_score"])

    return labeled, unlabeled


# -- Core prediction -----------------------------------------------------------

def _predict(scores: pd.Series, pos: float, neg: float) -> pd.Series:
    def _label(s: float) -> str:
        if s > pos:  return "positive"
        if s < neg:  return "negative"
        return "neutral"
    return scores.apply(_label)

def _accuracy(df: pd.DataFrame, pos: float, neg: float) -> float:
    pred = _predict(df["model_score"], pos, neg)
    return float((pred == df["human_label"]).mean())


# -- Threshold sweep -----------------------------------------------------------

def _sweep(df: pd.DataFrame) -> tuple[float, float]:
    """Return (optimal_symmetric_threshold, best_accuracy)."""
    best_acc    = 0.0
    best_thresh = SENTIMENT_POSITIVE_THRESHOLD
    for t_int in range(0, 51):          # 0.00 to 0.50 in 0.01 steps
        t   = round(t_int / 100, 2)
        acc = _accuracy(df, pos=t, neg=-t)
        if acc > best_acc:
            best_acc    = acc
            best_thresh = t
    return best_thresh, best_acc


# -- Confusion matrix ----------------------------------------------------------

def _print_confusion(df: pd.DataFrame, pos: float, neg: float) -> None:
    pred = _predict(df["model_score"], pos, neg)
    print()
    print(f"  {'':22}  Predicted")
    print(f"  {'':22}  {'pos':>10}  {'neu':>10}  {'neg':>10}  {'total':>7}")
    print(f"  {'':22}  {'---':>10}  {'---':>10}  {'---':>10}  {'-----':>7}")
    for true_lbl in LABEL_ORDER:
        mask      = df["human_label"] == true_lbl
        row_total = int(mask.sum())
        if row_total == 0:
            continue
        cells = []
        for pred_lbl in LABEL_ORDER:
            cnt = int(((pred == pred_lbl) & mask).sum())
            pct = cnt / row_total * 100
            marker = "◄" if pred_lbl == true_lbl else " "
            cells.append(f"{cnt:>4}({pct:>3.0f}%){marker}")
        print(f"  Actual {true_lbl[:3]:<14}  {'  '.join(cells)}  {row_total:>7}")
    print()
    correct = int((pred == df["human_label"]).sum())
    print(f"  Diagonal sum (correct): {correct} / {len(df)}  "
          f"= {correct/len(df):.1%}")


# -- Per-category accuracy -----------------------------------------------------

def _print_by_category(df: pd.DataFrame, pos: float, neg: float) -> None:
    if "category" not in df.columns:
        _info("No category column — export with main.py export-labels to get one")
        return

    pred = _predict(df["model_score"], pos, neg)
    df2  = df.copy()
    df2["_pred"]    = pred
    df2["_correct"] = df2["_pred"] == df2["human_label"]

    rows = []
    for cat, g in df2.groupby("category"):
        acc  = float(g["_correct"].mean())
        n    = len(g)
        wrong = g[~g["_correct"]]
        if len(wrong) > 0:
            mistakes = wrong.apply(
                lambda r: f"{r['human_label'][:3]}→{r['_pred'][:3]}", axis=1
            )
            top_mistake = Counter(mistakes).most_common(1)[0][0]
        else:
            top_mistake = "—"
        rows.append((acc, cat, n, top_mistake))

    rows.sort()   # worst accuracy first

    print()
    print(f"  {'Category':<22}  {'N':>4}  {'Accuracy':>9}  {'Top mistake'}")
    print(f"  {'-'*22}  {'-'*4}  {'-'*9}  {'-'*20}")
    for acc, cat, n, mistake in rows:
        bar  = "█" * int(acc * 15) + "░" * (15 - int(acc * 15))
        warn = " [!]" if acc < 0.60 else ""
        print(f"  {cat:<22}  {n:>4}  {acc:>8.1%}  {bar}  {mistake}{warn}")


# -- Holdout split validation --------------------------------------------------

def _holdout_report(df: pd.DataFrame) -> dict:
    """
    Chronological 60/40 tune/test split.
    Returns a dict of key metrics (or empty if too few labels).
    """
    MIN_HOLDOUT = 80
    if len(df) < MIN_HOLDOUT:
        _info(f"Holdout split needs ≥{MIN_HOLDOUT} unique labels "
              f"(have {len(df)} — {MIN_HOLDOUT - len(df)} more needed).")
        return {}

    # Sort chronologically so the test set is genuinely unseen
    sort_col = "published_at" if "published_at" in df.columns else None
    df2 = df.sort_values(sort_col).copy() if sort_col else df.copy()

    split    = int(len(df2) * 0.60)
    tune_df  = df2.iloc[:split]
    test_df  = df2.iloc[split:]

    _row("Tune set size", len(tune_df), "(60% — older headlines, used to tune)")
    _row("Test set size", len(test_df), "(40% — newer headlines, held out)")

    tuned_thresh, tune_acc = _sweep(tune_df)
    _row("Optimal threshold (tune set)", f"±{tuned_thresh:.2f}",
         f"tune accuracy = {tune_acc:.1%}")

    test_acc_tuned   = _accuracy(test_df, pos=tuned_thresh, neg=-tuned_thresh)
    test_acc_default = _accuracy(test_df,
                                 pos=SENTIMENT_POSITIVE_THRESHOLD,
                                 neg=SENTIMENT_NEGATIVE_THRESHOLD)

    _row("Test accuracy (tuned threshold)",   f"{test_acc_tuned:.1%}")
    _row("Test accuracy (current ±0.05)",     f"{test_acc_default:.1%}")

    gap = tune_acc - test_acc_tuned
    _row("Overfitting gap (tune − test)", f"{gap:+.1%}", warn=(gap > 0.10))

    if gap > 0.10:
        _warn("Large gap — threshold is overfit to current labels. Collect more before updating.")
    elif gap <= 0.03:
        _ok("Small gap — threshold generalizes well to held-out data.")
        if tuned_thresh != SENTIMENT_POSITIVE_THRESHOLD:
            _info(f"Consider updating SENTIMENT_POSITIVE_THRESHOLD = {tuned_thresh} "
                  f"in config.py (test accuracy: {test_acc_tuned:.1%})")
    else:
        _info("Moderate gap — keep labeling before committing a threshold change.")

    return {
        "tune_n":         len(tune_df),
        "test_n":         len(test_df),
        "tuned_threshold": float(tuned_thresh),
        "tune_accuracy":   float(tune_acc),
        "test_accuracy":   float(test_acc_tuned),
        "gap":             float(gap),
    }


# -- Label tracker (multi-file) ------------------------------------------------

def _tracker(search_dir: str = ".") -> None:
    _hdr("LABEL COLLECTION TRACKER")

    csvs = sorted(Path(search_dir).glob("labels_to_validate*.csv"))
    if not csvs:
        _warn("No labels_to_validate*.csv files found.")
        _info("Generate one with:  python main.py export-labels --n 300")
        _info("Then fill the human_label column and run:  python validate_labels.py <file>")
        return

    all_hashes: dict[str, str] = {}
    total_rows = 0

    for csv_path in csvs:
        try:
            df_raw = pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception:
            _warn(f"Could not read {csv_path.name}")
            continue

        if "human_label" not in df_raw.columns or "title" not in df_raw.columns:
            continue

        df_raw["human_label"] = df_raw["human_label"].astype(str).str.strip().str.lower()
        labeled = df_raw[df_raw["human_label"].isin(VALID_LABELS)]
        for _, row in labeled.iterrows():
            h = _normalise(str(row["title"]))[:80]
            if h not in all_hashes:
                all_hashes[h] = row["human_label"]
        total_rows += len(labeled)

    unique_n = len(all_hashes)

    _row("CSV files found", len(csvs))
    _row("Total labeled rows (all files)", total_rows)
    _row("Unique labeled headlines (deduped)", unique_n)

    print()
    for target in [198, TARGET_300, TARGET_500]:
        label = f"Baseline ({target})" if target == 198 else f"Target {target}"
        print(f"  {label:<16}  {_progress_bar(unique_n, target, unique_n)}")

    if all_hashes:
        dist = Counter(all_hashes.values())
        _sub("Label distribution (across all files, deduped)")
        for lbl in LABEL_ORDER:
            cnt = dist.get(lbl, 0)
            pct = cnt / unique_n * 100 if unique_n else 0
            _row(f"  {lbl}", cnt, f"({pct:.0f}%)")

        # Balance check
        counts = [dist.get(l, 0) for l in LABEL_ORDER]
        if max(counts) > 2 * min(c for c in counts if c > 0):
            _warn("Label distribution is imbalanced. "
                  "Try to label equal numbers of pos/neu/neg for reliable accuracy estimates.")
        else:
            _ok("Label distribution is reasonably balanced.")


# -- Save report ---------------------------------------------------------------

def _save_report(metrics: dict) -> str:
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    today     = date.today().isoformat()
    json_path = reports_dir / f"validate_{today}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    return str(json_path)


# -- Main ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the sentiment model against human-labeled headlines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("csv", nargs="?",
                        help="Labeled CSV path (from 'python main.py export-labels')")
    parser.add_argument("--threshold", type=float,
                        default=SENTIMENT_POSITIVE_THRESHOLD,
                        help=f"Score threshold to test (default: {SENTIMENT_POSITIVE_THRESHOLD})")
    parser.add_argument("--save", action="store_true",
                        help="Save metrics snapshot to reports/validate_YYYY-MM-DD.json")
    parser.add_argument("--tracker", action="store_true",
                        help="Show label collection progress only (scans current directory)")
    args = parser.parse_args()

    if args.tracker:
        _tracker()
        return

    if not args.csv:
        parser.error("Provide a labeled CSV path, or use --tracker to check progress.")

    _hdr(f"VALIDATION REPORT — {Path(args.csv).name}")

    try:
        labeled, unlabeled = _load(args.csv)
    except Exception as exc:
        _warn(f"Could not load CSV: {exc}")
        sys.exit(1)

    if labeled.empty:
        _warn("No rows with valid human_label found.")
        _info("Fill the 'human_label' column:  positive / neutral / negative")
        _info(f"Rows waiting for labels: {len(unlabeled)}")
        sys.exit(1)

    n = len(labeled)
    _row("Unique labeled headlines", n)
    _row("Rows still unlabeled", len(unlabeled))

    pos_thresh = args.threshold
    neg_thresh = -args.threshold

    # -- Label collection progress --------------------------------------------
    _sub("Label collection progress")
    for target in [TARGET_300, TARGET_500]:
        print(f"  Target {target:<5}  {_progress_bar(n, target, n)}")

    # -- Overall accuracy -----------------------------------------------------
    _sub(f"Overall accuracy at threshold ±{pos_thresh:.2f}")
    current_acc = _accuracy(labeled, pos_thresh, neg_thresh)
    _row("Overall accuracy", f"{current_acc:.1%}",
         f"({int(current_acc * n)}/{n} correct)")

    dist         = labeled["human_label"].value_counts()
    majority_acc = float(dist.max()) / n
    _row("Majority-class baseline", f"{majority_acc:.1%}",
         "(always predict the most common label)")
    improvement = current_acc - majority_acc
    _row("Improvement over baseline", f"{improvement:+.1%}",
         warn=(improvement < 0.05))
    if improvement >= 0.10:
        _ok("Model substantially outperforms the majority-class baseline.")
    elif improvement >= 0.0:
        _info("Model beats baseline but margin is small — watch for more labels.")
    else:
        _warn("Model is not outperforming the majority-class baseline.")

    # Per-label recall
    pred = _predict(labeled["model_score"], pos_thresh, neg_thresh)
    print()
    for lbl in LABEL_ORDER:
        mask = labeled["human_label"] == lbl
        if not mask.any():
            continue
        recall = float((pred[mask] == lbl).mean())
        n_lbl  = int(mask.sum())
        _row(f"  {lbl} recall", f"{recall:.1%}", f"(n={n_lbl})",
             warn=(recall < 0.50))

    # -- Threshold sweep -------------------------------------------------------
    _sub("Threshold sweep (symmetric, 0.00 – 0.50 in 0.01 steps)")
    opt_thresh, opt_acc = _sweep(labeled)
    _row("Current threshold", f"±{pos_thresh:.2f}",
         f"accuracy = {current_acc:.1%}")
    _row("Optimal threshold", f"±{opt_thresh:.2f}",
         f"accuracy = {opt_acc:.1%}")

    if opt_thresh == pos_thresh:
        _ok("Current threshold is already optimal for this label set.")
    else:
        delta = opt_acc - current_acc
        _info(f"Changing to ±{opt_thresh:.2f} would change accuracy by {delta:+.1%}.")
        if n < 200:
            _info("Collect ≥200 unique labels before committing a threshold change.")

    # -- Confusion matrix ------------------------------------------------------
    _sub("Confusion matrix  (actual rows × predicted columns)")
    _print_confusion(labeled, pos_thresh, neg_thresh)

    # -- Per-category accuracy -------------------------------------------------
    _sub("Per-category accuracy (worst first)")
    _print_by_category(labeled, pos_thresh, neg_thresh)

    # -- Holdout split ---------------------------------------------------------
    _sub("Holdout validation (chronological 60/40 tune/test split)")
    holdout_metrics = _holdout_report(labeled)

    # -- Save snapshot ----------------------------------------------------------
    metrics = {
        "date":               date.today().isoformat(),
        "csv":                str(args.csv),
        "n_labels":           n,
        "threshold":          float(pos_thresh),
        "overall_accuracy":   float(current_acc),
        "baseline_accuracy":  float(majority_acc),
        "improvement":        float(improvement),
        "optimal_threshold":  float(opt_thresh),
        "optimal_accuracy":   float(opt_acc),
        "holdout":            holdout_metrics,
    }

    if args.save:
        path = _save_report(metrics)
        print()
        _ok(f"Metrics snapshot saved → {path}")

    # -- Show tracker at end ---------------------------------------------------
    print()
    _tracker()


if __name__ == "__main__":
    main()
