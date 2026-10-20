import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from src.pipeline.rag import MultimodalRAG

SNIPPET_CHARS = 400


def _snippet(item: dict) -> str:
    content = (item.get("content") or "").strip()
    if not content:
        content = item.get("caption") or "[No text snippet available]"
    return " ".join(content.split())[:SNIPPET_CHARS]


def main():
    parser = argparse.ArgumentParser(description="Ask the multimodal RAG system.")
    parser.add_argument("question", type=str, help="Question to ask")
    parser.add_argument("--doc", action="append", help="Restrict retrieval to a document ID")
    parser.add_argument("--backend", default=None, help="Generation backend")
    parser.add_argument("--mode", default="multimodal", choices=["multimodal", "text_only", "no_rag"])
    parser.add_argument("--rerank", action="store_true", help="Force reranking on")
    parser.add_argument("--no-rerank", action="store_true", help="Force reranking off")
    parser.add_argument("--debug", action="store_true", help="Also print the raw model output")
    args = parser.parse_args()

    if args.rerank and args.no_rerank:
        parser.error("--rerank and --no-rerank are mutually exclusive")

    # None = follow retrieval.use_reranker in the config.
    use_reranker = True if args.rerank else (False if args.no_rerank else None)

    rag = MultimodalRAG.from_config(
        backend=args.backend,
        use_reranker=use_reranker,
    )

    out = rag.answer(
        args.question,
        doc_ids=args.doc,
        mode=args.mode,
    )

    print("\n=== ANSWER ===\n")
    print(out["answer"])

    notes = []
    if out.get("truncated"):
        notes.append(
            "generation hit the token limit; an unfinished final sentence "
            "was removed if present"
        )
    removed = (out.get("cleanup") or {}).get("removed_duplicates", 0)
    if removed:
        notes.append(f"{removed} repeated line(s)/sentence(s) removed")
    if notes:
        print("\n[Notes: " + "; ".join(notes) + ".]")

    if out.get("weak_evidence"):
        print("\n[Warning: retrieved evidence may be insufficient.]")

    contexts = out.get("contexts") or []

    if contexts and args.mode != "no_rag":
        status_by_number = {
            c["n"]: c.get("status") for c in (out.get("citations") or [])
        }

        print("\n=== RETRIEVED EVIDENCE (verification: none) ===")
        print(
            "Tags show which contexts the model referenced or a lexical "
            "heuristic suggested.\nNeither checks that a source supports a claim."
        )

        for i, item in enumerate(contexts, start=1):
            status = status_by_number.get(i)
            tag = ""
            if status == "model_referenced_unverified":
                tag = "   <- referenced by model (unverified)"
            elif status == "suggested_lexical":
                tag = "   <- lexical suggestion (not verification)"

            print(
                f"\n[{i}] {item.get('doc_id', 'unknown document')} | "
                f"page {item.get('page_num', '?')} | "
                f"{item.get('modality', 'unknown')}{tag}"
            )
            print(_snippet(item))

    if args.debug:
        print("\n=== RAW MODEL OUTPUT ===\n")
        print(out.get("raw_answer", ""))

    latency = out.get("latency")
    if latency is not None:
        print(f"\n=== LATENCY ===\n{latency}")

    stats = out.get("generation_stats")
    if stats:
        print(f"generation_stats: {stats}")


if __name__ == "__main__":
    main()