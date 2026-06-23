"""
Label-quality tooling (methodological improvements, 2026-06-19).

Covers what the scorer benchmark does NOT: the trustworthiness of the human
ground truth itself.

Subcommands
-----------
disagreements <labeled.csv>
    Pull every model-vs-human mismatch into disagreements_to_review.csv with a
    blank `verdict` column. You tag each: was the model right, were you (the
    human) right, or is it genuinely ambiguous?

disagreements <reviewed.csv> --summary
    Read the filled file back. Reports the TRUE error rate — some "model errors"
    are the model being right and the human being inconsistent, which means real
    accuracy is HIGHER than the raw agreement number.

consistency-export <labeled.csv> [--n 50]
    Sample N already-labeled headlines, strip ALL labels (model + human), write
    consistency_relabel.csv for blind re-labeling.

consistency-check <original.csv> <relabel.csv>
    Join on id, report intra-annotator agreement = how often you agree with your
    PAST self. That is the ceiling the scorer is graded against: if you only
    agree with yourself 85% of the time, an 83% scorer is essentially maxed out.

Usage
-----
    python label_audit.py disagreements labels_validated.csv
    python label_audit.py disagreements disagreements_to_review.csv --summary
    python label_audit.py consistency-export labels_validated.csv --n 50
    python label_audit.py consistency-check labels_validated.csv consistency_relabel.csv
"""

import argparse
import sys

import pandas as pd

_LABELS = ["positive", "neutral", "negative"]


def _norm(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _confusion(rows_truth: pd.Series, cols_pred: pd.Series, row_name: str, col_name: str) -> None:
    print(f"  (rows = {row_name}, cols = {col_name})")
    print(f"  {'':>10}" + "".join(f"{c:>10}" for c in _LABELS))
    for t in _LABELS:
        mask = rows_truth == t
        counts = [int(((cols_pred == p) & mask).sum()) for p in _LABELS]
        print(f"  {t:>10}" + "".join(f"{c:>10}" for c in counts))


# -- disagreements -----------------------------------------------------------------

def cmd_disagreements(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.csv)

    if args.summary:
        if "verdict" not in df.columns:
            print("  No 'verdict' column — is this the reviewed file?")
            return
        v = _norm(df["verdict"])
        done = v.isin(["model", "human", "ambiguous"])
        n = int(done.sum())
        if n == 0:
            print("  No verdicts filled yet. Tag each row: model | human | ambiguous")
            return
        model_right = int((v == "model").sum())
        human_right = int((v == "human").sum())
        ambiguous   = int((v == "ambiguous").sum())
        print(f"  Adjudicated {n} disagreements:")
        print(f"    model was right (you drifted) : {model_right}")
        print(f"    human was right (real error)  : {human_right}")
        print(f"    genuinely ambiguous           : {ambiguous}")
        print()
        print("  Interpretation:")
        print(f"    - {model_right} of the {n} 'errors' weren't model errors — real")
        print("      scorer accuracy is HIGHER than the raw agreement number.")
        print(f"    - {ambiguous} ambiguous cases are the irreducible label-noise floor.")
        print(f"    - {human_right} are genuine model errors worth a rubric/few-shot fix.")
        return

    df["human_label"] = _norm(df["human_label"])
    df["model_label"] = _norm(df["model_label"])
    df = df[df["human_label"].isin(_LABELS)]
    dis = df[df["model_label"] != df["human_label"]].copy()
    dis["verdict"] = ""   # model | human | ambiguous

    cols = [c for c in ["id", "category", "title", "model_label", "model_score",
                        "model_relevance", "human_label", "verdict"] if c in dis.columns]
    out = "disagreements_to_review.csv"
    dis[cols].to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  {len(dis)} model-vs-human disagreements -> {out}")
    print("  Fill 'verdict' for each: model (model right) | human (you right) | ambiguous")
    print("  Then: python label_audit.py disagreements disagreements_to_review.csv --summary")


# -- consistency -------------------------------------------------------------------

def cmd_consistency_export(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.csv)
    df = df[_norm(df["human_label"]).isin(_LABELS)]
    if df.empty:
        print("  No labeled rows found in that file.")
        return
    samp = df.sample(min(args.n, len(df)), random_state=7)[["id", "title"]].copy()
    samp["human_label"] = ""        # blind: model labels and your originals hidden
    samp["human_relevant"] = ""
    out = "consistency_relabel.csv"
    samp.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  {len(samp)} headlines to re-label BLIND -> {out}")
    print("  Re-label without looking at the original file.")
    print("  For a clean signal, wait ~2 weeks so you don't just recall your answers.")
    print("  Then: python label_audit.py consistency-check "
          f"{args.csv} {out}")


def cmd_consistency_check(args: argparse.Namespace) -> None:
    orig = pd.read_csv(args.original)
    relabel = pd.read_csv(args.relabel)
    m = (orig[["id", "human_label"]]
         .merge(relabel[["id", "human_label"]], on="id", suffixes=("_orig", "_new")))
    a = _norm(m["human_label_orig"])
    b = _norm(m["human_label_new"])
    valid = b.isin(_LABELS) & a.isin(_LABELS)
    a, b = a[valid], b[valid]
    if len(a) == 0:
        print("  No overlapping re-labeled rows. Fill consistency_relabel.csv first.")
        return
    agree = float((a.values == b.values).mean())
    print(f"  Intra-annotator agreement: {agree:.1%}  (n={len(a)})")
    print(f"  => This is the practical ceiling for scorer accuracy on this task.")
    if agree < 0.85:
        print("     Below 85%: your own convention is noisy — tighten the rubric")
        print("     before chasing scorer gains.")
    else:
        print("     >=85%: labels are stable; scorer headroom is real if it lags.")
    print()
    _confusion(a.reset_index(drop=True), b.reset_index(drop=True), "original", "re-label")


def main() -> None:
    p = argparse.ArgumentParser(description="Label-quality audit tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("disagreements", help="Model-vs-human mismatch review + summary")
    d.add_argument("csv")
    d.add_argument("--summary", action="store_true")
    d.set_defaults(func=cmd_disagreements)

    e = sub.add_parser("consistency-export", help="Blind re-label set for intra-annotator check")
    e.add_argument("csv")
    e.add_argument("--n", type=int, default=50)
    e.set_defaults(func=cmd_consistency_export)

    c = sub.add_parser("consistency-check", help="Compare original vs re-label")
    c.add_argument("original")
    c.add_argument("relabel")
    c.set_defaults(func=cmd_consistency_check)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
