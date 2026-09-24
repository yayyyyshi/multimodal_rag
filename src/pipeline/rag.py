"""End-to-end question answering, used by ask.py, the Streamlit app and evaluate.py.

    rag = MultimodalRAG.from_config()
    out = rag.answer("What share of Latinos expect their children to be better off?")
    out["answer"], out["citations"]

Modes: "multimodal" retrieves text, tables and images; "text_only" retrieves
plain text only; "no_rag" asks the LLM without any context.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from src.generation.generator import get_generator, supports_images
from src.generation.prompts import ABSTAIN, build_no_rag_messages, build_rag_messages
from src.retrieval.retriever import MultimodalRetriever
from src.utils.config import load_config, resolve
from src.vectorstore.chroma_store import ChromaStore

CITE_RE = re.compile(r"\[(\d+)\]")


def is_abstention(answer: str) -> bool:
    a = answer.strip().lower()
    return a.startswith(ABSTAIN.lower()) or a in {"", "unknown", "i don't know", "i do not know"}


class MultimodalRAG:
    def __init__(self, cfg: dict, store: ChromaStore, manifest: dict, backend: str | None = None,
                 use_reranker: bool | None = None):
        self.cfg = cfg
        self.store = store
        self.manifest = manifest
        self.artifacts_dir = resolve(cfg["paths"]["artifacts_dir"])
        self.retriever = MultimodalRetriever(store, manifest, cfg, use_reranker=use_reranker)
        g = cfg["generation"]
        self.backend = backend or g["backend"]
        model = {"transformers": cfg["models"]["generator"], "transformers_vl": cfg["models"]["vl_generator"],
                 "ollama": g["ollama_model"]}.get(self.backend)
        self.generator = get_generator(self.backend, model, g["max_new_tokens"], g["temperature"], g["ollama_url"])

    @classmethod
    def from_config(cls, config_path=None, overrides: dict | None = None, backend: str | None = None,
                    use_reranker: bool | None = None) -> "MultimodalRAG":
        cfg = load_config(config_path, overrides)
        art = resolve(cfg["paths"]["artifacts_dir"])
        manifest_path = art / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"{manifest_path} not found. Build the artifacts on Kaggle and unzip them into {art}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        store = ChromaStore(resolve(cfg["paths"]["chroma_dir"]))
        return cls(cfg, store, manifest, backend, use_reranker)

    def _load_images(self, items: list[dict]) -> dict[int, object]:
        """Returns {context_number: PIL.Image} for up to max_images figures. Empty for text-only backends."""
        n_max = self.cfg["generation"].get("max_images", 0)
        if not supports_images(self.backend) or n_max <= 0:
            return {}
        from PIL import Image
        out = {}
        for i, it in enumerate(items, 1):
            p = it.get("image_path")
            if it.get("modality") == "image" and p and (self.artifacts_dir / p).exists():
                out[i] = Image.open(self.artifacts_dir / p).convert("RGB")
                if len(out) >= n_max:
                    break
        return out

    def answer(self, question: str, doc_ids: list[str] | None = None, mode: str = "multimodal",
               k: int | None = None) -> dict:
        t0 = time.perf_counter()
        items, weak, retrieval = [], False, None
        if mode == "no_rag":
            messages, images = build_no_rag_messages(question), {}
        else:
            retrieval = self.retriever.retrieve(question, doc_ids=doc_ids, k=k, mode=mode)
            items, weak = retrieval.items, retrieval.weak_evidence
            images = self._load_images(items)
            messages = build_rag_messages(question, items, weak=weak, image_refs=images)
        t1 = time.perf_counter()

        if mode != "no_rag" and self.cfg["generation"].get("abstain_policy") == "hard" and weak:
            answer = ABSTAIN
        else:
            answer = self.generator.generate(messages, images=list(images.values()) or None)
        t2 = time.perf_counter()

        cited = sorted({int(n) for n in CITE_RE.findall(answer) if 1 <= int(n) <= len(items)})
        citations = [{
            "n": n, "doc_id": items[n - 1]["doc_id"], "page": items[n - 1]["page_num"],
            "modality": items[n - 1]["modality"], "image_path": items[n - 1].get("image_path") or None,
            "snippet": (items[n - 1]["content"] or "")[:300],
        } for n in cited]
        return {
            "question": question, "mode": mode, "answer": answer, "abstained": is_abstention(answer),
            "citations": citations, "contexts": items, "weak_evidence": weak,
            "best_text_score": retrieval.best_text_score if retrieval else None,
            "latency": {"retrieval_s": round(t1 - t0, 3), "generation_s": round(t2 - t1, 3)},
        }
