"""Bucket answerable questions from predictions_<mode>.jsonl by failure type.

Run from the project root:
    python scripts/analyze_failures.py results/e2e_text_v1/predictions_text_only.jsonl
    python scripts/analyze_failures.py <file> --show 12      # print N examples per bucket

Buckets (answerable questions only):
    correct                 score says correct
    refused_gold_in_top     model abstained although a gold page was retrieved
    refused_gold_missing    model abstained and no gold page was retrieved
    wrong_gold_in_top       model answered, gold page retrieved, score says wrong
    wrong_gold_missing      model answered, no gold page retrieved, score says wrong

Only gold *pages* are known here; passage text is not saved by evaluate.py, so
"page retrieved" does not prove the answer text was in the passage.
The gold-vs-prediction printout lets you spot answers that are right but
marked wrong (a scoring problem rather than a model problem).
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_correct(score):
    """score_answer's return type is unknown: accept bool, number or dict."""
    if isinstance(score, dict):
        for key in ("correct", "is_correct", "score", "acc", "accuracy"):
            if key in score:
                return is_correct(score[key])
        return False
    if isinstance(score, bool):
        return score
    if isinstance(score, (int, float)):
        return score >= 0.5
    return False


def retrieved_pages(row):
    pages = []
    for tag in row.get("retrieved") or []:
        m = re.match(r"p(-?\d+):", tag)
        if m:
            pages.append(int(m.group(1)))
    return pages


def as_pages(value):
    out = []
    for v in value or []:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--show", type=int, default=6, help="examples printed per bucket")
    args = ap.parse_args()

    rows = load(args.predictions)
    if rows:
        r0 = rows[0]
        print(f"[info] {len(rows)} rows; score type={type(r0.get('score')).__name__} "
              f"example={r0.get('score')!r}; keys={sorted(r0)}")

    buckets = defaultdict(list)
    shift_hits = 0  # gold page found only if pages are shifted by one
    for r in rows:
        if not r.get("answerable"):
            continue
        gold = set(as_pages(r.get("evidence_pages")))
        got = set(retrieved_pages(r))
        in_top = bool(gold & got)
        if not in_top and (gold & {p + 1 for p in got} or gold & {p - 1 for p in got}):
            shift_hits += 1
        if is_correct(r.get("score")):
            name = "correct"
        elif r.get("abstained"):
            name = "refused_gold_in_top" if in_top else "refused_gold_missing"
        else:
            name = "wrong_gold_in_top" if in_top else "wrong_gold_missing"
        buckets[name].append(r)

    total = sum(len(v) for v in buckets.values())
    print(f"\nAnswerable questions: {total}\n")
    for name in ("correct", "refused_gold_in_top", "refused_gold_missing",
                 "wrong_gold_in_top", "wrong_gold_missing"):
        print(f"{name:<24}{len(buckets[name]):>4}")
    print(f"\nGold page missing but off by one page: {shift_hits} "
          "(non-zero would suggest a page-numbering offset)")

    fmts = defaultdict(lambda: [0, 0])
    for name, items in buckets.items():
        for r in items:
            fmts[r.get("answer_format")][0] += 1
            fmts[r.get("answer_format")][1] += name == "correct"
    print("\nAccuracy by gold answer_format (n, correct):")
    for fmt, (n, c) in sorted(fmts.items(), key=lambda x: str(x[0])):
        print(f"  {str(fmt):<10}{n:>4}{c:>4}")

    for name in ("wrong_gold_in_top", "refused_gold_in_top", "wrong_gold_missing"):
        print(f"\n===== {name} (first {args.show}) =====")
        for r in buckets[name][: args.show]:
            print(f"\nqid={r.get('qid')} fmt={r.get('answer_format')} "
                  f"gold_pages={r.get('evidence_pages')} retrieved={r.get('retrieved')}")
            print("Q:   ", r.get("question"))
            print("GOLD:", r.get("answer"))
            print("PRED:", " ".join(str(r.get("prediction", "")).split())[:300])


if __name__ == "__main__":
    main()
