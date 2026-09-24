"""Step 4: evaluate on the MMLongBench-Doc subset.

    python scripts/evaluate.py --retrieval-only       # no LLM, fine on CPU
    python scripts/evaluate.py --modes no_rag text_only multimodal --backend transformers_vl
    python scripts/evaluate.py --limit 30              # first 30 questions only

Retrieval is filtered to the question's own document, as in the benchmark.
--global searches all documents instead.

Output in results/<run_name>/: predictions_<mode>.jsonl, summary.json, summary.md
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metrics import breakdown_by_evidence, page_metrics, score_answer, summarize
from src.utils.config import resolve
from src.utils.metadata import load_jsonl, save_jsonl

KEYS = [("accuracy", "Accuracy"), ("accuracy_answerable", "Acc. (answerable)"),
        ("hallucination_rate", "Hallucination rate"), ("hallucination_rate_unanswerable", "Halluc. on unanswerable"),
        ("over_refusal_rate_answerable", "Over-refusal"), ("ret_hit@1", "Hit@1"), ("ret_hit@5", "Hit@5"),
        ("ret_recall@5", "Recall@5"), ("ret_mrr", "MRR"), ("avg_latency_s", "Latency (s)")]


def fmt(v):
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def to_markdown(summary: dict, breakdown: dict) -> str:
    modes = list(summary)
    lines = ["## Overall", "", "| Metric | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for k, label in KEYS:
        if any(k in summary[m] for m in modes):
            lines.append(f"| {label} | " + " | ".join(fmt(summary[m].get(k)) for m in modes) + " |")
    for m in modes:
        lines += ["", f"## By evidence type: {m}", "", "| Evidence | n | Acc. | Hit@5 | Recall@5 | MRR |", "|---|---|---|---|---|---|"]
        for ev, s in breakdown[m].items():
            lines.append(f"| {ev} | {s['n']} | {fmt(s.get('accuracy'))} | {fmt(s.get('ret_hit@5'))} | "
                         f"{fmt(s.get('ret_recall@5'))} | {fmt(s.get('ret_mrr'))} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=["text_only", "multimodal"],
                    choices=["no_rag", "text_only", "multimodal"])
    ap.add_argument("--retrieval-only", action="store_true", help="skip the LLM, only retrieval metrics")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--global", dest="global_search", action="store_true", help="search all documents")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--questions", default="data/eval/questions.jsonl")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    questions = load_jsonl(resolve(args.questions))
    if args.limit:
        questions = questions[: args.limit]
    run = args.run_name or datetime.now().strftime("%Y%m%d_%H%M") + ("_retrieval" if args.retrieval_only else "")
    out_dir = resolve("results") / run
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.retrieval_only:
        from src.retrieval.retriever import MultimodalRetriever
        from src.utils.config import load_config
        from src.vectorstore.chroma_store import ChromaStore
        cfg = load_config(args.config)
        manifest = json.loads((resolve(cfg["paths"]["artifacts_dir"]) / "manifest.json").read_text())
        retriever = MultimodalRetriever(ChromaStore(resolve(cfg["paths"]["chroma_dir"])), manifest, cfg,
                                        use_reranker=not args.no_rerank)
        rag = None
    else:
        from src.pipeline.rag import MultimodalRAG
        rag = MultimodalRAG.from_config(args.config, backend=args.backend, use_reranker=not args.no_rerank)

    indexed = set(rag.store.list_docs() if rag else retriever.store.list_docs())
    skipped = [q for q in questions if q["doc_id"] not in indexed]
    questions = [q for q in questions if q["doc_id"] in indexed]
    if skipped:
        print(f"[eval] skipping {len(skipped)} questions whose document is not indexed")

    summary, breakdown = {}, {}
    modes = [m for m in args.modes if not (args.retrieval_only and m == "no_rag")]
    for mode in modes:
        rows, t0 = [], time.time()
        for i, q in enumerate(questions, 1):
            doc_ids = None if args.global_search else [q["doc_id"]]
            row = {k: q[k] for k in ("qid", "doc_id", "question", "answer", "answer_format",
                                     "evidence_pages", "evidence_sources", "answerable")}
            if args.retrieval_only:
                ts = time.perf_counter()
                res = retriever.retrieve(q["question"], doc_ids=doc_ids, mode=mode)
                row["latency_s"] = time.perf_counter() - ts
                items = res.items
            else:
                out = rag.answer(q["question"], doc_ids=doc_ids, mode=mode)
                items = out["contexts"]
                row.update(prediction=out["answer"], abstained=out["abstained"],
                           score=score_answer(out["answer"], q["answer"], q["answer_format"]),
                           latency_s=sum(out["latency"].values()),
                           cited_pages=[c["page"] for c in out["citations"]])
            if mode != "no_rag":
                pages = [it["page_num"] for it in items if (args.global_search is False or it["doc_id"] == q["doc_id"])]
                row["retrieved"] = [f"p{it['page_num']}:{it['modality']}" for it in items]
                row["retrieval"] = page_metrics(pages, q["evidence_pages"]) if q["answerable"] else None
            rows.append(row)
            if i % 20 == 0 or i == len(questions):
                print(f"[eval:{mode}] {i}/{len(questions)} ({time.time() - t0:.0f}s)")
        save_jsonl(rows, out_dir / f"predictions_{mode}.jsonl")
        summary[mode] = summarize(rows)
        breakdown[mode] = breakdown_by_evidence(rows)

    (out_dir / "summary.json").write_text(json.dumps({"summary": summary, "by_evidence": breakdown}, indent=2))
    md = to_markdown(summary, breakdown)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md + f"\n[eval] results -> {out_dir}")


if __name__ == "__main__":
    main()
