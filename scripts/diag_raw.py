"""Read-only diagnostic: raw generation, Ollama timings, post-processing effect."""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.prompts import build_rag_messages
from src.pipeline.rag import MultimodalRAG, remove_repeated_bullets


def ns_to_s(value):
    return round(value / 1e9, 2) if value else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--doc", action="append")
    p.add_argument("--mode", default="text_only")
    p.add_argument("--runs", type=int, default=1)
    args = p.parse_args()

    rag = MultimodalRAG.from_config(backend="ollama", use_reranker=False)

    t0 = time.perf_counter()
    res = rag.retriever.retrieve(args.question, doc_ids=args.doc, mode=args.mode)
    print(f"retrieval_s = {time.perf_counter() - t0:.3f}  weak={res.weak_evidence}  "
          f"best_text={res.best_text_score:.3f}")

    print("\n=== RETRIEVED ITEMS ===")
    for i, it in enumerate(res.items, 1):
        body = " ".join((it.get("content") or it.get("caption") or "").split())
        print(f"[{i}] p.{it.get('page_num')} {it.get('modality')} "
              f"len={len(body)} text_score={it.get('text_score')} rrf={round(it.get('rrf', 0), 4)}")
        print(f"     {body[:160]}")

    messages = build_rag_messages(args.question, res.items, weak=res.weak_evidence)
    print(f"\nsystem_chars={len(messages[0]['content'])}  user_chars={len(messages[1]['content'])}")

    gen = rag.generator
    payload = {"model": gen.model, "messages": messages, "stream": False, "options": gen.options}
    print("options =", gen.options)

    for run in range(1, args.runs + 1):
        req = urllib.request.Request(
            f"{gen.url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        wall = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(r.read().decode("utf-8"))
        wall = time.perf_counter() - wall

        ev, evd = out.get("eval_count"), out.get("eval_duration")
        pe, ped = out.get("prompt_eval_count"), out.get("prompt_eval_duration")
        print(f"\n=== RUN {run} TIMINGS ===")
        print(f"wall_s={wall:.2f} total_s={ns_to_s(out.get('total_duration'))} "
              f"load_s={ns_to_s(out.get('load_duration'))}")
        print(f"prompt_tokens={pe} prompt_eval_s={ns_to_s(ped)} "
              f"prompt_tok_per_s={round(pe / (ped / 1e9), 1) if pe and ped else None}")
        print(f"gen_tokens={ev} gen_s={ns_to_s(evd)} "
              f"gen_tok_per_s={round(ev / (evd / 1e9), 1) if ev and evd else None}")
        print(f"done_reason={out.get('done_reason')}   (length = hit num_predict cap)")

        raw = out["message"]["content"].strip()
        print(f"\n=== RAW MODEL OUTPUT (repr) ===\n{raw!r}")
        print(f"\n=== RAW (rendered) ===\n{raw}")
        print(f"\n=== AFTER remove_repeated_bullets (repr) ===\n{remove_repeated_bullets(raw)!r}")


if __name__ == "__main__":
    main()