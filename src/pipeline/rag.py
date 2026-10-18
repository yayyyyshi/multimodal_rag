
"""End-to-end multimodal RAG pipeline with conservative evidence suggestions."""

from __future__ import annotations

import json
import re
import time
from difflib import SequenceMatcher

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.generation.generator import get_generator, supports_images
from src.generation.prompts import (
    ABSTAIN,
    build_no_rag_messages,
    build_rag_messages,
)
from src.retrieval.retriever import MultimodalRetriever
from src.utils.config import load_config, resolve
from src.vectorstore.chroma_store import ChromaStore


CITE_RE = re.compile(r"\[(\d+)\]")
NUMBERED_LINE_RE = re.compile(r"^\s*\d+[.)]\s+")
SUGGESTION_RE = re.compile(
    r"\s*\(suggested evidence:\s*\[\d+\]\)", re.IGNORECASE
)

MAX_NUMBERED_BULLETS = 4
SUGGESTION_THRESHOLD = 0.22


def is_abstention(answer: str) -> bool:
    """Return True when the model abstains or returns no answer."""
    text = answer.strip().lower()
    return text.startswith(ABSTAIN.lower()) or text in {
        "",
        "unknown",
        "i don't know",
        "i do not know",
    }


def _normalise_bullet(text: str) -> str:
    """Normalize a bullet for duplicate comparison."""
    text = NUMBERED_LINE_RE.sub("", text).strip()
    text = SUGGESTION_RE.sub("", text)
    text = CITE_RE.sub("", text)
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _is_duplicate_bullet(candidate: str, previous: list[str]) -> bool:
    """Detect exact and near-duplicate numbered bullets."""
    candidate_norm = _normalise_bullet(candidate)
    if not candidate_norm:
        return False

    candidate_tokens = set(candidate_norm.split())

    for old_norm in previous:
        if candidate_norm == old_norm:
            return True

        old_tokens = set(old_norm.split())
        if not candidate_tokens or not old_tokens:
            continue

        jaccard = len(candidate_tokens & old_tokens) / len(
            candidate_tokens | old_tokens
        )
        sequence_ratio = SequenceMatcher(
            None, candidate_norm, old_norm
        ).ratio()

        if jaccard >= 0.55 or sequence_ratio >= 0.78:
            return True

    return False


def remove_repeated_bullets(
    answer: str,
    max_bullets: int = MAX_NUMBERED_BULLETS,
) -> str:
    """Remove duplicate numbered bullets and renumber the remaining list.

    Non-numbered lines are preserved. This is formatting cleanup, not
    factual verification, and cannot restore content the model omitted.
    """
    if not answer or not answer.strip():
        return answer.strip()

    lines = answer.splitlines()
    output: list[str] = []
    bullets: list[str] = []
    normalized_bullets: list[str] = []
    in_numbered_list = False
    inserted_bullets = False

    for line in lines:
        stripped = line.strip()

        if not NUMBERED_LINE_RE.match(stripped):
            output.append(line)
            continue

        in_numbered_list = True
        content = NUMBERED_LINE_RE.sub("", stripped).strip()

        if _is_duplicate_bullet(content, normalized_bullets):
            continue

        if len(bullets) >= max_bullets:
            continue

        bullets.append(content)
        normalized_bullets.append(_normalise_bullet(content))

    if not in_numbered_list:
        return answer.strip()

    # Rebuild the cleaned list where the first numbered item appeared.
    rebuilt: list[str] = []

    for line in output:
        if NUMBERED_LINE_RE.match(line.strip()):
            if not inserted_bullets:
                rebuilt.extend(
                    f"{index}. {bullet}"
                    for index, bullet in enumerate(bullets, start=1)
                )
                inserted_bullets = True
            continue

        rebuilt.append(line)

    # Insert the list if the original numbered items were removed.
    if not inserted_bullets and bullets:
        rebuilt.extend(
            f"{index}. {bullet}"
            for index, bullet in enumerate(bullets, start=1)
        )

    cleaned = "\n".join(rebuilt).strip()
    return cleaned or answer.strip()


def suggest_citations(
    answer: str,
    items: list[dict],
    threshold: float = SUGGESTION_THRESHOLD,
) -> tuple[str, list[int]]:
    """Append labeled TF-IDF evidence suggestions to numbered bullets.

    These are lexical matches only. They do not prove that a source
    supports a claim and must not be treated as verified citations.
    """
    if not items or is_abstention(answer):
        return answer, []

    contexts = [
        (item.get("content") or item.get("caption") or "").strip()
        for item in items
    ]
    valid_indices = [
        index for index, content in enumerate(contexts) if content
    ]

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

    candidate_texts = [
        NUMBERED_LINE_RE.sub("", line)
        for _, line in candidates
    ]
    corpus = [contexts[index] for index in valid_indices]
    corpus.extend(candidate_texts)

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        vectors = vectorizer.fit_transform(corpus)
        context_vectors = vectors[:len(valid_indices)]
        answer_vectors = vectors[len(valid_indices):]
        scores = cosine_similarity(answer_vectors, context_vectors)
    except ValueError:
        # Empty vocabulary or no usable lexical features.
        return answer, []

    suggested_numbers: set[int] = set()

    for row, (line_index, line) in enumerate(candidates):
        # Preserve citations already supplied by the model.
        if CITE_RE.search(line):
            continue

        best_position = int(scores[row].argmax())
        best_score = float(scores[row, best_position])

        if best_score < threshold:
            continue

        context_index = valid_indices[best_position]
        source_number = context_index + 1
        suggested_numbers.add(source_number)

        lines[line_index] = (
            f"{line} (suggested evidence: [{source_number}])"
        )

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

        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        store = ChromaStore(resolve(cfg["paths"]["chroma_dir"]))

        return cls(
            cfg,
            store,
            manifest,
            backend,
            use_reranker,
        )

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
    def _valid_citation_numbers(
        answer: str,
        item_count: int,
    ) -> set[int]:
        """Return citation numbers referring to retrieved contexts."""
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

        if (
            mode != "no_rag"
            and self.cfg["generation"].get("abstain_policy") == "hard"
            and weak
        ):
            answer_text = ABSTAIN
        else:
            # Exactly one generation call per query.
            answer_text = self.generator.generate(
                messages,
                images=list(images.values()) or None,
            )

        suggested_numbers: list[int] = []

        if mode != "no_rag" and not is_abstention(answer_text):
            answer_text = remove_repeated_bullets(answer_text)

            if items:
                answer_text, suggested_numbers = suggest_citations(
                    answer_text,
                    items,
                )

        generation_end = time.perf_counter()

        valid_numbers = self._valid_citation_numbers(
            answer_text,
            len(items),
        )

        displayed_numbers = sorted(
            valid_numbers | set(suggested_numbers)
        )

        citations = [
            {
                "n": number,
                "doc_id": items[number - 1]["doc_id"],
                "page": items[number - 1]["page_num"],
                "modality": items[number - 1]["modality"],
                "image_path": (
                    items[number - 1].get("image_path") or None
                ),
                "snippet": (
                    items[number - 1].get("content")
                    or items[number - 1].get("caption")
                    or ""
                )[:300],
                "suggested": number in suggested_numbers,
            }
            for number in displayed_numbers
        ]

        return {
            "question": question,
            "mode": mode,
            "answer": answer_text,
            "abstained": is_abstention(answer_text),
            "citations": citations,
            "contexts": items,
            "weak_evidence": weak,
            "best_text_score": (
                retrieval.best_text_score if retrieval else None
            ),
            "latency": {
                "retrieval_s": round(
                    retrieval_end - start_time, 3
                ),
                "generation_s": round(
                    generation_end - retrieval_end, 3
                ),
            },
        }
