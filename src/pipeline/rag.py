"""End-to-end multimodal RAG pipeline.

Source attribution is reported in separate, honestly labelled categories.
None of them is verified against the answer text:

  contexts          everything retrieved and shown to the model;
  model-referenced  [n] markers the model wrote (unverified claims);
  suggested         optional TF-IDF lexical matches; disabled by default
                    (generation.suggest_citations: true to enable).
"""

from __future__ import annotations

import json
import re
import time

from src.generation.generator import get_generator, supports_images
from src.generation.prompts import (
    ABSTAIN,
    build_no_rag_messages,
    build_rag_messages,
)
from src.pipeline.postprocess import CITE_RE, clean_answer, normalise_abstention
from src.retrieval.retriever import MultimodalRetriever
from src.utils.config import load_config, resolve
from src.vectorstore.chroma_store import ChromaStore


NUMBERED_LINE_RE = re.compile(r"^\s*\d+[.)]\s+")
SUGGESTION_THRESHOLD = 0.22

STATUS_MODEL = "model_referenced_unverified"
STATUS_SUGGESTED = "suggested_lexical"


def is_abstention(answer: str) -> bool:
    """Return True when the model abstains or returns no answer."""
    text = answer.strip().lower()
    return text.startswith(ABSTAIN.lower()) or text in {
        "",
        "unknown",
        "i don't know",
        "i do not know",
    }


def remove_repeated_bullets(answer: str, max_bullets: int | None = None) -> str:
    """Backward-compatible wrapper around postprocess.clean_answer().

    max_bullets is ignored: the old silent cap was removed because it could
    discard correct content. Formatting only; not factual verification.
    """
    return clean_answer(answer)[0]


def suggest_citations(
    answer: str,
    items: list[dict],
    threshold: float = SUGGESTION_THRESHOLD,
) -> tuple[str, list[int]]:
    """Append labeled TF-IDF evidence suggestions to numbered bullets.

    These are lexical matches only. They do not prove that a source
    supports a claim and must not be treated as verified citations.
    Disabled by default; requires scikit-learn only when enabled.
    """
    if not items or is_abstention(answer):
        return answer, []

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    contexts = [
        (item.get("content") or item.get("caption") or "").strip()
        for item in items
    ]
    valid_indices = [i for i, content in enumerate(contexts) if content]

    if not valid_indices:
        return answer, []

    lines = answer.splitlines()
    candidates = [
        (line_index, line.strip())
        for line_index, line in enumerate(lines)
        if NUMBERED_LINE_RE.match(line.strip())
    ]

    if not candidates:
        return answer, []

    candidate_texts = [NUMBERED_LINE_RE.sub("", line) for _, line in candidates]
    corpus = [contexts[i] for i in valid_indices] + candidate_texts

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        vectors = vectorizer.fit_transform(corpus)
        scores = cosine_similarity(
            vectors[len(valid_indices):],
            vectors[:len(valid_indices)],
        )
    except ValueError:
        return answer, []

    suggested_numbers: set[int] = set()

    for row, (line_index, line) in enumerate(candidates):
        if CITE_RE.search(line):
            continue

        best_position = int(scores[row].argmax())
        if float(scores[row, best_position]) < threshold:
            continue

        source_number = valid_indices[best_position] + 1
        suggested_numbers.add(source_number)
        lines[line_index] = f"{line} (suggested evidence: [{source_number}])"

    return "\n".join(lines), sorted(suggested_numbers)


class MultimodalRAG:
    def __init__(
        self,
        cfg: dict,
        store: ChromaStore,
        manifest: dict,
        backend: str | None = None,
        use_reranker: bool | None = None,
    ):
        self.cfg = cfg
        self.store = store
        self.manifest = manifest
        self.artifacts_dir = resolve(cfg["paths"]["artifacts_dir"])

        self.retriever = MultimodalRetriever(
            store,
            manifest,
            cfg,
            use_reranker=use_reranker,
        )

        generation = cfg["generation"]
        self.backend = backend or generation["backend"]

        model = {
            "transformers": cfg["models"]["generator"],
            "transformers_vl": cfg["models"]["vl_generator"],
            "ollama": generation["ollama_model"],
        }.get(self.backend)

        self.generator = get_generator(
            self.backend,
            model,
            generation["max_new_tokens"],
            generation["temperature"],
            generation["ollama_url"],
        )

    @classmethod
    def from_config(
        cls,
        config_path=None,
        overrides: dict | None = None,
        backend: str | None = None,
        use_reranker: bool | None = None,
    ) -> "MultimodalRAG":
        cfg = load_config(config_path, overrides)
        artifacts_dir = resolve(cfg["paths"]["artifacts_dir"])
        manifest_path = artifacts_dir / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found. Build the artifacts on Kaggle "
                f"and unzip them into {artifacts_dir}"
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        store = ChromaStore(resolve(cfg["paths"]["chroma_dir"]))

        return cls(cfg, store, manifest, backend, use_reranker)

    def _load_images(self, items: list[dict]) -> dict[int, object]:
        """Load images only when the configured generator supports them."""
        generation = self.cfg["generation"]
        max_images = generation.get("max_images", 0)

        if not supports_images(self.backend) or max_images <= 0:
            return {}

        if self.backend == "ollama":
            model_name = generation["ollama_model"].lower()
            if "vl" not in model_name:
                return {}

        from PIL import Image

        images: dict[int, object] = {}

        for index, item in enumerate(items, start=1):
            image_path = item.get("image_path")

            if (
                item.get("modality") == "image"
                and image_path
                and (self.artifacts_dir / image_path).exists()
            ):
                with Image.open(self.artifacts_dir / image_path) as image:
                    images[index] = image.convert("RGB")

                if len(images) >= max_images:
                    break

        return images

    @staticmethod
    def _valid_citation_numbers(answer: str, item_count: int) -> set[int]:
        """Return [n] markers that refer to a retrieved context."""
        return {
            int(number)
            for number in CITE_RE.findall(answer)
            if 1 <= int(number) <= item_count
        }

    def answer(
        self,
        question: str,
        doc_ids: list[str] | None = None,
        mode: str = "multimodal",
        k: int | None = None,
    ) -> dict:
        start_time = time.perf_counter()

        items: list[dict] = []
        weak = False
        retrieval = None
        images: dict[int, object] = {}

        if mode == "no_rag":
            messages = build_no_rag_messages(question)
        else:
            retrieval = self.retriever.retrieve(
                question,
                doc_ids=doc_ids,
                k=k,
                mode=mode,
            )

            items = retrieval.items
            weak = retrieval.weak_evidence
            images = self._load_images(items)

            messages = build_rag_messages(
                question,
                items,
                weak=weak,
                image_refs=images,
            )

        retrieval_end = time.perf_counter()
        generation_cfg = self.cfg["generation"]
        generation_stats: dict = {}

        if (
            mode != "no_rag"
            and generation_cfg.get("abstain_policy") == "hard"
            and weak
        ):
            raw_answer = ABSTAIN
        else:
            # Exactly one generation call per query.
            raw_answer = self.generator.generate(
                messages,
                images=list(images.values()) or None,
            )
            generation_stats = dict(
                getattr(self.generator, "last_stats", None) or {}
            )

        done_reason = generation_stats.get("done_reason")
        truncated = (done_reason == "length") if done_reason else None

        answer_text = normalise_abstention(raw_answer, ABSTAIN)
        cleanup = {"removed_duplicates": 0, "trimmed_incomplete": False}

        if mode != "no_rag" and not is_abstention(answer_text):
            answer_text, cleanup = clean_answer(
                answer_text,
                truncated=bool(truncated),
            )

        # [n] markers the model actually wrote in the displayed answer.
        # Computed before any suggestion text is appended.
        model_numbers = self._valid_citation_numbers(answer_text, len(items))

        suggested_numbers: list[int] = []
        if (
            mode != "no_rag"
            and items
            and generation_cfg.get("suggest_citations", False)
            and not is_abstention(answer_text)
        ):
            answer_text, suggested_numbers = suggest_citations(
                answer_text,
                items,
            )

        generation_end = time.perf_counter()

        citations = []
        for number in sorted(model_numbers | set(suggested_numbers)):
            item = items[number - 1]
            from_model = number in model_numbers
            citations.append({
                "n": number,
                "doc_id": item["doc_id"],
                "page": item["page_num"],
                "modality": item["modality"],
                "image_path": item.get("image_path") or None,
                "snippet": (
                    item.get("content") or item.get("caption") or ""
                )[:300],
                "status": STATUS_MODEL if from_model else STATUS_SUGGESTED,
                "suggested": not from_model,
            })

        return {
            "question": question,
            "mode": mode,
            "answer": answer_text,
            "raw_answer": raw_answer,
            "abstained": is_abstention(answer_text),
            # Entries are model-referenced (unverified) or lexical
            # suggestions; see each entry's "status".
            "citations": citations,
            "verification": "none",
            "contexts": items,
            "weak_evidence": weak,
            "best_text_score": (
                retrieval.best_text_score if retrieval else None
            ),
            "truncated": truncated,
            "cleanup": cleanup,
            "generation_stats": generation_stats,
            "latency": {
                "retrieval_s": round(retrieval_end - start_time, 3),
                "generation_s": round(generation_end - retrieval_end, 3),
            },
        }