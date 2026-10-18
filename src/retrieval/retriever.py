
"""Find relevant text, tables and images with weighted RRF and reranking."""

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


def rrf_fuse(
    ranked_lists: list[tuple[list[dict], float]],
    k: int = 60,
) -> list[dict]:
    """Weighted reciprocal rank fusion."""
    fused: dict[str, dict] = {}

    for items, weight in ranked_lists:
        for rank, item in enumerate(items, start=1):
            chunk_id = item["chunk_id"]
            entry = fused.setdefault(
                chunk_id, {**item, "rrf": 0.0, "hits": []}
            )
            entry["rrf"] += weight / (k + rank)
            entry["hits"].append(item.get("_src", "?"))

    return sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)


def diversify_results(
    items: list[dict],
    k: int,
    max_per_page: int = 2,
) -> list[dict]:
    """Prefer distinct documents/pages while retaining relevant candidates."""
    selected = []
    deferred = []
    page_counts: dict[tuple[str, str], int] = {}
    doc_counts: dict[str, int] = {}

    def identity(item: dict) -> tuple[str, str, str]:
        metadata = item.get("metadata") or {}
        doc_id = str(item.get("doc_id") or metadata.get("doc_id") or "")
        page = str(
            item.get("page_num")
            or item.get("page")
            or metadata.get("page_num")
            or metadata.get("page")
            or ""
        )
        return doc_id, page, str(item.get("chunk_id", ""))

    # First pass: allow at most two chunks per page and prefer documents
    # not already represented when scores are comparable.
    for item in items:
        doc_id, page, _ = identity(item)
        page_key = (doc_id, page)

        if page_counts.get(page_key, 0) >= max_per_page:
            deferred.append(item)
            continue

        selected.append(item)
        page_counts[page_key] = page_counts.get(page_key, 0) + 1
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

        if len(selected) >= k:
            return selected

    # Fill remaining slots with deferred items if necessary.
    for item in deferred:
        selected.append(item)
        if len(selected) >= k:
            break

    return selected


class MultimodalRetriever:
    def __init__(
        self,
        store,
        manifest: dict,
        cfg: dict,
        use_reranker: bool | None = None,
    ):
        self.store = store
        self.rcfg = cfg["retrieval"]

        self.text_model = get_text_embedder(
            manifest["text_embedder"],
            manifest.get("text_query_instruction", ""),
        )
        self.clip = get_clip(manifest["clip"])

        reranker_name = cfg["models"].get("reranker")
        self.reranker = None

        if reranker_name and (use_reranker is None or use_reranker):
            try:
                self.reranker = get_reranker(reranker_name)
            except Exception as exc:
                print(f"[retriever] reranker disabled: {exc}")

    def retrieve(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        k: int | None = None,
        mode: str = "multimodal",
    ) -> RetrievalResult:
        cfg = self.rcfg
        k = k or cfg["final_k"]

        query_vector = self.text_model.embed_query(query)

        if mode == "text_only":
            text_hits = self.store.query_text(
                query_vector,
                cfg["top_k_text"],
                doc_ids,
                modality="text",
            )
            image_hits = []
        else:
            text_hits = self.store.query_text(
                query_vector, cfg["top_k_text"], doc_ids
            )
            image_hits = self.store.query_image(
                self.clip.embed_query(query),
                cfg["top_k_image"],
                doc_ids,
            )

        for hit in text_hits:
            hit["_src"] = "text"
            hit["text_score"] = hit["score"]

        for hit in image_hits:
            hit["_src"] = "clip"
            hit["clip_score"] = hit["score"]

        fused = rrf_fuse(
            [
                (text_hits, 1.0),
                (image_hits, cfg["image_weight"]),
            ],
            cfg["rrf_k"],
        )

        best_text = max(
            (hit["score"] for hit in text_hits),
            default=0.0,
        )

        best_rerank = None
        candidate_count = len(fused)

        if self.reranker and fused:
            candidates = fused[: max(k * 3, 15)]
            scores = self.reranker.score(
                query,
                [candidate.get("content", "") for candidate in candidates],
            )

            for candidate, score in zip(candidates, scores):
                candidate["rerank"] = float(score)

            fused = sorted(
                candidates,
                key=lambda candidate: candidate.get("rerank", float("-inf")),
                reverse=True,
            )
            best_rerank = max(scores) if len(scores) else None

        # Diversify after ranking so one page's figures do not dominate.
        items = diversify_results(fused, k)

        for rank, item in enumerate(items, start=1):
            item["rank"] = rank
            item.pop("_src", None)

        weak = best_text < cfg["min_text_score"]
        if best_rerank is not None:
            weak = weak or best_rerank < cfg["min_rerank_score"]

        return RetrievalResult(
            items=items,
            weak_evidence=weak,
            best_text_score=best_text,
            best_rerank_score=best_rerank,
            debug={
                "n_text_hits": len(text_hits),
                "n_image_hits": len(image_hits),
                "n_fused_candidates": candidate_count,
                "n_final_items": len(items),
            },
        )
