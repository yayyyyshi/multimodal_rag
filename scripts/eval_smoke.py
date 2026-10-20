"""Small repeatable evaluation set for the Germanwings document.

Run:  python scripts/eval_smoke.py --repeats 2

IMPORTANT: the automatic checks are heuristics (substring and regex tests
on the answer text). They catch formatting, abstention, truncation,
repetition and a few known factual traps. They do NOT verify that an
answer is faithful to the evidence. Read each printed answer yourself and
compare it with the retrieved passages (saved in the JSON output).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import re
import statistics

DEFAULT_DOC = "germanwingsdigitalcrisisanalysis_150403064828_conversion_gate01_95"

# Subject-attached negation: "... Twitter and Facebook were inaccessible".
# Deliberately does not match "Twitter and Facebook remained available
# while the website was inaccessible".
TWITTER_FB_DOWN = (
    r"(twitter|facebook)( and (twitter|facebook))?,? "
    r"(were|was|remained|became|are|is) (also )?"
    r"(inaccessible|unavailable|not available|down)"
)

CASES = [
    {
        "id": "single_fact",
        "category": "answerable from one passage",
        "question": "How long was the Germanwings website inaccessible after the crash?",
        "expect": "answer",
        "include_any": [["2 hours", "two hours", "approx. 2", "approximately 2"]],
        "forbid": [],
    },
    {
        "id": "multi_passage",
        "category": "needs several passages",
        "question": "How did Germanwings communicate updates during the first hours of the crisis?",
        "expect": "answer",
        "include_any": [["tweet", "twitter"], ["facebook"],
                        ["crisis site", "crisis website", "fifth hour", "hour 5", "5th hour"]],
        "forbid": [TWITTER_FB_DOWN],
    },
    {
        # VERIFY before trusting: this fact must be absent from the document.
        # Check with:  Select-String -Path <path>\chunks.jsonl -Pattern "passengers"
        "id": "unanswerable",
        "category": "unanswerable in-domain",
        "question": "What was the total number of passengers Germanwings carried in 2014?",
        "expect": "abstain",
        "include_any": [],
        "forbid": [],
    },
    {
        "id": "confusable",
        "category": "two passages easy to confuse",
        "question": ("During the first hour, were Twitter and Facebook available to the public "
                     "while the Germanwings website was down?"),
        "expect": "answer",
        "include_any": [["available", "only available", "remained"]],
        "forbid": [TWITTER_FB_DOWN],
    },
    {
        # Control for yes-bias: the correct answer is the opposite of the
        # "confusable" case, so echoing the question's premise fails.
        "id": "confusable_neg",
        "category": "negated control (detects premise echoing)",
        "question": "Was the Germanwings website accessible during the first 60 minutes after the incident?",
        "expect": "answer",
        "include_any": [["inaccessible", "not accessible", "crash", "unavailable", "not available", "was down"]],
        "forbid": [r"^\s*yes\b", r"(website|site) (was|remained) (accessible|available)"],
    },
    {
        "id": "repetition_prone",
        "category": "previously repetitive output",
        "question": "What were the main findings of the Germanwings digital crisis analysis?",
        "expect": "answer",
        "include_any": [],
        "forbid": [],
    },
]


SHORT_OK = {"a", "an", "as", "at", "be", "by", "do", "he", "if", "in", "is", "it", "no",
            "of", "on", "or", "so", "to", "up", "we", "us", "me", "my", "am", "go", "i"}
_CITE = re.compile(r"\[\d+\]")
NOISE_LIMIT = 0.10  # fitted on a handful of real answers; a smoke check, not a classifier


def noise_ratio(text: str) -> float:
    """Share of tokens that look like OCR junk (odd symbols, unexplained 1-2 letter bits)."""
    tokens = _CITE.sub(" ", text).split()
    if not tokens:
        return 0.0
    bad = 0
    for token in tokens:
        core = token.strip(".,;:!?()\"'")
        if not core or re.fullmatch(r"[^\w]+", core) or re.search(r"[\\|=<>~_]", core):
            bad += 1
        elif core.isdigit() or re.fullmatch(r"[\d.,:/%$-]+", core):
            continue
        elif len(core) <= 2 and core.lower() not in SHORT_OK and not core.isupper():
            bad += 1
    return bad / len(tokens)


def looks_like_slide_header(text: str) -> bool:
    letters = [c for c in _CITE.sub("", text) if c.isalpha()]
    return len(letters) >= 8 and sum(c.isupper() for c in letters) / len(letters) > 0.8


def check_run(case: dict, out: dict) -> dict:
    """Heuristic checks for one answer. Returns flag -> bool."""
    answer = out["answer"]
    low = answer.lower()
    cleanup = out.get("cleanup") or {}
    abstained = bool(out.get("abstained"))

    flags = {
        "format_ok": (
            answer == answer.strip()
            and "**" not in answer
            and "\n\n\n" not in answer
        ),
        "not_truncated": out.get("truncated") is not True,
        "no_repetition": cleanup.get("removed_duplicates", 0) == 0,
        "abstain_ok": abstained == (case["expect"] == "abstain"),
        # Catches answers that copy OCR garbage or a slide title verbatim.
        "readable": noise_ratio(answer) < NOISE_LIMIT and not looks_like_slide_header(answer),
    }

    if case["expect"] == "abstain":
        flags["includes_ok"] = True  # abstain_ok already covers this case
    else:
        flags["includes_ok"] = (not abstained) and all(
            any(alt in low for alt in group) for group in case["include_any"]
        )

    flags["no_known_error"] = not any(
        re.search(pattern, low) for pattern in case["forbid"]
    )
    return flags


def run_cases(rag, cases, doc, repeats, mode, k=None):
    results = []

    for case in cases:
        for run in range(1, repeats + 1):
            out = rag.answer(case["question"], doc_ids=[doc], mode=mode, k=k)
            flags = check_run(case, out)
            stats = out.get("generation_stats") or {}

            results.append({
                "case": case["id"],
                "category": case["category"],
                "run": run,
                "model": getattr(rag.generator, "model", None),
                "question": case["question"],
                "answer": out["answer"],
                "raw_answer": out.get("raw_answer"),
                "abstained": out.get("abstained"),
                "flags": flags,
                "passed": all(flags.values()),
                "model_referenced": [c["n"] for c in out.get("citations", [])],
                "retrieved": [
                    {"n": i, "page": it.get("page_num"), "modality": it.get("modality"),
                     "chars": len((it.get("content") or it.get("caption") or "")),
                     "content": (it.get("content") or it.get("caption") or "")}
                    for i, it in enumerate(out.get("contexts", []), start=1)
                ],
                "weak_evidence": out.get("weak_evidence"),
                "latency": out.get("latency"),
                "generation_stats": stats,
            })

            r = results[-1]
            print(f"\n--- {case['id']} (run {run}) : {'PASS' if r['passed'] else 'FAIL'} ---")
            print("Q:", case["question"])
            print("A:", out["answer"])
            print("flags:", {k: v for k, v in flags.items()})
            referenced = set(r["model_referenced"])
            print("retrieved passages (* = model wrote this [n]; unverified):")
            for item in r["retrieved"]:
                mark = "*" if item["n"] in referenced else " "
                snippet = " ".join(item["content"].split())[:100]
                print(f"  {mark}[{item['n']}] p.{item['page']} {item['modality']} {item['chars']}ch  {snippet}")
            print("tokens: prompt=%s gen=%s done=%s | latency=%s"
                  % (stats.get("prompt_tokens"), stats.get("gen_tokens"),
                     stats.get("done_reason"), out.get("latency")))
            print("ollama stats:", stats)

    return results


def summarise(results):
    print("\n=== SUMMARY (heuristic checks only; read the answers) ===")
    header = (f"{'case':<18}{'runs':>5}{'pass':>6}  {'ret_s(avg)':>10}{'gen_s(avg)':>11}"
              f"{'gen_tok(avg)':>13}{'prompt_tok':>11}")
    print(header)

    for case_id in dict.fromkeys(r["case"] for r in results):
        rows = [r for r in results if r["case"] == case_id]
        ret_s = [r["latency"]["retrieval_s"] for r in rows if r.get("latency")]
        gen_s = [r["latency"]["generation_s"] for r in rows if r.get("latency")]
        gen_t = [r["generation_stats"].get("gen_tokens") for r in rows if r["generation_stats"].get("gen_tokens")]
        p_t = [r["generation_stats"].get("prompt_tokens") for r in rows if r["generation_stats"].get("prompt_tokens")]
        print(f"{case_id:<18}{len(rows):>5}{sum(r['passed'] for r in rows):>6}  "
              f"{(statistics.mean(ret_s) if ret_s else float('nan')):>10.1f}"
              f"{(statistics.mean(gen_s) if gen_s else float('nan')):>11.1f}"
              f"{(statistics.mean(gen_t) if gen_t else float('nan')):>13.0f}"
              f"{(statistics.mean(p_t) if p_t else float('nan')):>11.0f}")

    total = len(results)
    print(f"\noverall: {sum(r['passed'] for r in results)}/{total} runs passed all heuristic checks")


def main():
    parser = argparse.ArgumentParser(description="Smoke evaluation for the RAG pipeline.")
    parser.add_argument("--doc", default=DEFAULT_DOC)
    parser.add_argument("--repeats", type=int, default=2,
                        help="Runs per case (answers vary between runs)")
    parser.add_argument("--mode", default="text_only", choices=["multimodal", "text_only"])
    parser.add_argument("--rerank", action="store_true", help="Enable the reranker (default: off)")
    parser.add_argument("--k", type=int, default=None,
                        help="Contexts passed to the model (default: retrieval.final_k)")
    parser.add_argument("--cases", nargs="*", help="Case ids to run (default: all)")
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--ollama-model", default=None,
                        help="Override generation.ollama_model for this run (no config edit needed)")
    parser.add_argument("--out", default="eval_results.json")
    args = parser.parse_args()

    from src.pipeline.rag import MultimodalRAG

    cases = [c for c in CASES if not args.cases or c["id"] in args.cases]
    rag = MultimodalRAG.from_config(
        backend=args.backend,
        use_reranker=True if args.rerank else False,
    )

    if args.ollama_model:
        if args.backend != "ollama":
            parser.error("--ollama-model only applies to --backend ollama")
        from src.generation.generator import get_generator

        gen_cfg = rag.cfg["generation"]
        gen_cfg["ollama_model"] = args.ollama_model
        rag.generator = get_generator(
            "ollama",
            args.ollama_model,
            gen_cfg["max_new_tokens"],
            gen_cfg["temperature"],
            gen_cfg["ollama_url"],
        )

    print(f"generator: backend={args.backend} "
          f"model={getattr(rag.generator, 'model', 'n/a')} "
          f"rerank={'on' if args.rerank else 'off'} k={args.k or 'default'}")

    results = run_cases(rag, cases, args.doc, args.repeats, args.mode, k=args.k)
    summarise(results)

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull results (answers, raw answers, retrieved items, stats): {args.out}")


if __name__ == "__main__":
    main()