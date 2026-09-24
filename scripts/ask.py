"""Ask one question from the terminal.

    python scripts/ask.py "How many Latinos see their children being less well off?"
    python scripts/ask.py "..." --doc PH_2016_06_08_Economy_Final   # search one document
    python scripts/ask.py "..." --backend ollama
    python scripts/ask.py "..." --mode no_rag                       # LLM without retrieval
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.rag import MultimodalRAG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--doc", nargs="*", default=None, help="doc_id(s) to search in")
    ap.add_argument("--mode", default="multimodal", choices=["multimodal", "text_only", "no_rag"])
    ap.add_argument("--backend", default=None, choices=["transformers", "transformers_vl", "ollama", "extractive"])
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    rag = MultimodalRAG.from_config(args.config, backend=args.backend, use_reranker=not args.no_rerank)
    out = rag.answer(args.question, doc_ids=args.doc, mode=args.mode)

    print("\n=== ANSWER ===\n" + out["answer"])
    if out["weak_evidence"]:
        print("(retrieval evidence was weak)")
    print("\n=== SOURCES ===")
    for c in out["citations"]:
        print(f"[{c['n']}] {c['doc_id']} p.{c['page']} ({c['modality']})" + (f"  {c['image_path']}" if c["image_path"] else ""))
    if not out["citations"] and out["contexts"]:
        print("(no citation in answer) top retrieved:")
        for it in out["contexts"][:3]:
            print(f"  - {it['doc_id']} p.{it['page_num']} ({it['modality']})")
    print(f"\nlatency: {out['latency']}")


if __name__ == "__main__":
    main()
