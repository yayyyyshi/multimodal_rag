

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
import argparse

from src.pipeline.rag import MultimodalRAG


def main():
    parser = argparse.ArgumentParser(description="Ask the multimodal RAG system.")
    parser.add_argument("question", type=str, help="Question to ask")
    parser.add_argument("--doc", action="append", help="Restrict retrieval to a document ID")
    parser.add_argument("--backend", default=None, help="Generation backend")
    parser.add_argument("--mode", default="multimodal", choices=["multimodal", "text_only", "no_rag"])
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    args = parser.parse_args()

    rag = MultimodalRAG.from_config(
        backend=args.backend,
        use_reranker=False if args.no_rerank else None,
    )

    out = rag.answer(
        args.question,
        doc_ids=args.doc,
        mode=args.mode,
    )

    print("\n=== ANSWER ===\n")
    print(out["answer"])

    if out.get("weak_evidence"):
        print("\n[Warning: retrieved evidence may be insufficient.]")

    citations = out.get("citations") or []
    contexts = out.get("contexts") or []

    if citations:
        print("\n=== CITED SOURCES ===")
        for c in citations:
            print(
                f"[{c['n']}] {c['doc_id']} "
                f"p.{c['page']} ({c['modality']})"
            )
            if c.get("snippet"):
                print(f"    {c['snippet'].replace(chr(10), ' ')}")
    elif contexts and args.mode != "no_rag":
        print("\n=== RETRIEVED EVIDENCE (not claim-level citations) ===")
        for i, item in enumerate(contexts, start=1):
            doc_id = item.get("doc_id", "unknown document")
            page = item.get("page_num", "?")
            modality = item.get("modality", "unknown")
            content = (item.get("content") or "").strip()
            if not content:
                content = item.get("caption") or "[No text snippet available]"
            content = " ".join(content.split())

            print(f"\n[{i}] {doc_id} | page {page} | {modality}")
            print(content[:700])

    latency = out.get("latency")
    if latency is not None:
        print(f"\n=== LATENCY ===\n{latency}")


if __name__ == "__main__":
    main()

