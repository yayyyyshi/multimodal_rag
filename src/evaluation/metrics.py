"""Retrieval and answer metrics.

Retrieval is scored at page level against the gold evidence pages: Hit@k,
Recall@k and MRR. Answers are scored by rules instead of the GPT-4 judge used
by the benchmark authors: numbers with a 1% tolerance, strings by normalized
containment or token F1, lists by the share of items found.

A hallucination is either an answer to an unanswerable question or a wrong
answer that was not an abstention.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict

ABSTAIN = "not answerable"
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def page_metrics(retrieved_pages: list[int], gold_pages: list[int], ks=(1, 3, 5)) -> dict:
    """retrieved_pages is the page of each retrieved chunk in rank order; repeats are allowed."""
    gold = set(gold_pages)
    out = {}
    if not gold:
        return out
    uniq = list(dict.fromkeys(retrieved_pages))
    for k in ks:
        top = retrieved_pages[:k]
        out[f"hit@{k}"] = float(any(p in gold for p in top))
        out[f"recall@{k}"] = len(gold & set(top)) / len(gold)
    rr = 0.0
    for i, p in enumerate(uniq, 1):
        if p in gold:
            rr = 1.0 / i
            break
    out["mrr"] = rr
    return out


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\[\d+\]", " ", s)  # remove [n] citations
    s = re.sub(r"[^a-z0-9%.\- ]", " ", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip(" .")


def main_answer(pred: str) -> str:
    """First line of the prediction, where the prompt asks for the short answer."""
    first = pred.strip().split("\n")[0]
    return first if len(first) < 400 else first[:400]


def _numbers(s: str) -> list[float]:
    out = []
    for m in NUM_RE.findall(s):
        try:
            out.append(float(m.replace(",", "").rstrip(".")))
        except ValueError:
            pass
    return out


def _num_match(gold: float, preds: list[float], rel=0.01) -> bool:
    for p in preds:
        for g in (gold, gold * 100, gold / 100):
            if abs(p - g) <= max(abs(g) * rel, 1e-6) or abs(p - g) < 0.011:
                return True
    return False


def token_f1(a: str, b: str) -> float:
    ta, tb = normalize(a).split(), normalize(b).split()
    if not ta or not tb:
        return 0.0
    common = sum(min(ta.count(t), tb.count(t)) for t in set(ta))
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def _as_list(gold: str) -> list[str]:
    try:
        v = ast.literal_eval(gold)
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
    except Exception:
        pass
    return [x.strip() for x in re.split(r",|;| and ", gold) if x.strip()]


def is_abstain(pred: str) -> bool:
    return normalize(pred).startswith(ABSTAIN) or normalize(pred) in {"", "unknown", "i don t know"}


def score_answer(pred: str, gold: str, answer_format: str) -> float:
    """Returns 1.0 or 0.0; List answers can score in between."""
    if normalize(gold) == ABSTAIN or answer_format == "None":
        return float(is_abstain(pred))
    if is_abstain(pred):
        return 0.0
    ans = main_answer(pred)
    if answer_format in ("Int", "Float"):
        g = _numbers(gold)
        return float(bool(g) and _num_match(g[0], _numbers(ans)))
    if answer_format == "List":
        items = _as_list(gold)
        if not items:
            return 0.0
        hit = 0
        for it in items:
            gi = normalize(it)
            gn = _numbers(it)
            if (gi and gi in normalize(pred)) or (gn and _num_match(gn[0], _numbers(pred))) or token_f1(it, pred) > 0.8:
                hit += 1
        return hit / len(items)
    g, p = normalize(gold), normalize(ans)
    if g and (g in p or (p and p in g and len(p) >= 0.5 * len(g))):
        return 1.0
    gn = _numbers(gold)
    if gn and len(g) < 12 and _num_match(gn[0], _numbers(ans)):
        return 1.0
    return float(token_f1(gold, ans) >= 0.5)


def summarize(rows: list[dict]) -> dict:
    """Each row needs 'answerable'; 'score', 'abstained' and 'retrieval' are used when present."""
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    s = {"n": len(rows), "n_answerable": len(ans), "n_unanswerable": len(una)}
    if rows and "score" in rows[0]:
        s["accuracy"] = sum(r["score"] for r in rows) / len(rows)
        s["accuracy_answerable"] = sum(r["score"] for r in ans) / max(len(ans), 1)
        s["abstain_correct_unanswerable"] = sum(r["abstained"] for r in una) / max(len(una), 1)
        s["hallucination_rate_unanswerable"] = 1 - s["abstain_correct_unanswerable"]
        wrong_committed = [r for r in ans if not r["abstained"] and r["score"] < 0.5]
        s["wrong_answer_rate_answerable"] = len(wrong_committed) / max(len(ans), 1)
        s["over_refusal_rate_answerable"] = sum(r["abstained"] for r in ans) / max(len(ans), 1)
        s["hallucination_rate"] = (len(wrong_committed) + sum(not r["abstained"] for r in una)) / max(len(rows), 1)
        lat = [r.get("latency_s", 0) for r in rows]
        s["avg_latency_s"] = sum(lat) / max(len(lat), 1)
    ret = [r["retrieval"] for r in ans if r.get("retrieval")]
    if ret:
        for key in ret[0]:
            s[f"ret_{key}"] = sum(x[key] for x in ret) / len(ret)
    return s


def breakdown_by_evidence(rows: list[dict]) -> dict[str, dict]:
    groups = defaultdict(list)
    for r in rows:
        if not r["answerable"]:
            groups["Unanswerable"].append(r)
            continue
        for src in r.get("evidence_sources") or ["?"]:
            groups[src].append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}
