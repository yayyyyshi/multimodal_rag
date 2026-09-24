"""Finds the most relevant text, tables and images for a question.

The question is embedded twice: with the text model to search mm_text, and
with the CLIP text encoder to search mm_image. The two ranked lists are merged
with weighted Reciprocal Rank Fusion and optionally reranked by a
cross-encoder.

weak_evidence is set when the best text similarity (or rerank score) is below
the configured threshold; the generator is then told to abstain if unsure.

mode "text_only" searches plain-text chunks only. It is the baseline used in
the evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.embeddings.models import get_clip, get_reranker, get_text_embedder


@dataclass
class RetrievalResult:
    items: list[dict]
    weak_evidence: bool
    best_text_score: float
    best_rerank_score: float | None = None
    debug: dict = field(default_factory=dict)


def rrf_fuse(ranked_lists: list[tuple[list[dict], float]], k: int = 60) -> list[dict]:
    """Weighted RRF: score = sum of weight / (k + rank). Each list is (items, weight)."""
    fused: dict[str, dict] = {}
    for items, weight in ranked_lists:
        for rank, it in enumerate(items):
            cid = it["chunk_id"]
            entry = fused.setdefault(cid, {**it, "rrf": 0.0, "hits": []})
            entry["rrf"] += weight / (k + rank + 1)
            entry["hits"].append(it.get("_src", "?"))
    return sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)


class MultimodalRetriever:
    def __init__(self, store, manifest: dict, cfg: dict, use_reranker: bool | None = None):
        self.store = store
        self.rcfg = cfg["retrieval"]
        # query models come from the manifest so they match the stored vectors
        self.text_model = get_text_embedder(manifest["text_embedder"], manifest.get("text_query_instruction", ""))
        self.clip = get_clip(manifest["clip"])
        rer = cfg["models"].get("reranker")
        self.reranker = None
        if rer and (use_reranker is None or use_reranker):
            try:
                self.reranker = get_reranker(rer)
            except Exception as e:  # e.g. no internet to download the model
                print(f"[retriever] reranker disabled: {e}")

    def retrieve(self, query: str, doc_ids: list[str] | None = None, k: int | None = None,
                 mode: str = "multimodal") -> RetrievalResult:
        r = self.rcfg
        k = k or r["final_k"]
        qv = self.text_model.embed_query(query)

        if mode == "text_only":
            text_hits = self.store.query_text(qv, r["top_k_text"], doc_ids, modality="text")
            image_hits = []
        else:
            text_hits = self.store.query_text(qv, r["top_k_text"], doc_ids)
            image_hits = self.store.query_image(self.clip.embed_query(query), r["top_k_image"], doc_ids)

        for h in text_hits:
            h["_src"], h["text_score"] = "text", h["score"]
        for h in image_hits:
            h["_src"], h["clip_score"] = "clip", h["score"]

        fused = rrf_fuse([(text_hits, 1.0), (image_hits, r["image_weight"])], r["rrf_k"])
        best_text = max((h["score"] for h in text_hits), default=0.0)

        best_rerank = None
        if self.reranker and fused:
            cands = fused[: max(k * 2, 10)]
            scores = self.reranker.score(query, [c["content"] for c in cands])
            for c, s in zip(cands, scores):
                c["rerank"] = s
            fused = sorted(cands, key=lambda x: x["rerank"], reverse=True)
            best_rerank = max(scores) if scores else None

        items = fused[:k]
        for i, it in enumerate(items, 1):
            it["rank"] = i
            it.pop("_src", None)

        weak = best_text < r["min_text_score"]
        if best_rerank is not None:
            weak = weak or best_rerank < r["min_rerank_score"]
        return RetrievalResult(items, weak, best_text, best_rerank,
                               debug={"n_text_hits": len(text_hits), "n_image_hits": len(image_hits)})
