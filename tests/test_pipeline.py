"""Tests that run without a GPU or model downloads, using the hash:<dim> embedders.

    pytest -q
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunking.chunker import chunk_document, split_long_text  # noqa: E402
from src.evaluation.metrics import page_metrics, score_answer, summarize  # noqa: E402
from src.preprocessing.cleaner import apply_ocr_fixes, clean_blocks  # noqa: E402
from src.retrieval.retriever import rrf_fuse  # noqa: E402

SAMPLE_PDF = ROOT / "data" / "raw" / "check.pdf"


# cleaning
def test_ocr_fixes_only_touch_numbers():
    assert apply_ocr_fixes("Revenue 1O5 million") == "Revenue 105 million"
    assert apply_ocr_fixes("address 0x1F and 5G") == "address 0x1F and 5G"   # the old l->I / 0->O rules changed these
    assert apply_ocr_fixes("let l be the length") == "let l be the length"


def test_clean_blocks_drops_noise_and_running_headers():
    blocks = []
    for p in range(1, 6):
        blocks += [
            {"doc_id": "d", "page_num": p, "text": "ACME RESEARCH CENTER", "source_type": "pymupdf_text"},
            {"doc_id": "d", "page_num": p, "text": str(p), "source_type": "pymupdf_text"},
            {"doc_id": "d", "page_num": p, "text": f"Real content on page {p}.", "source_type": "pymupdf_text"},
        ]
    cleaned, _ = clean_blocks(blocks)
    texts = [b["text"] for b in cleaned]
    assert "ACME RESEARCH CENTER" not in texts
    assert all(not t.isdigit() for t in texts)
    assert len(cleaned) == 5


# chunking
def test_split_long_text_respects_max():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    parts = split_long_text(text, 300, 50)
    assert len(parts) > 5 and all(len(p) <= 300 + 60 for p in parts)


def test_chunks_never_cross_pages_and_tables_keep_header():
    table = "| a | b |\n|---|---|\n" + "\n".join(f"| {i} | {i*2} |" for i in range(60))
    blocks = [
        {"doc_id": "d", "page_num": 1, "text": "Title of the doc", "source_type": "t", "modality": "text"},
        {"doc_id": "d", "page_num": 1, "text": "x " * 800, "source_type": "t", "modality": "text"},
        {"doc_id": "d", "page_num": 2, "text": "page two text", "source_type": "t", "modality": "text"},
        {"doc_id": "d", "page_num": 2, "text": table, "source_type": "pymupdf_table", "modality": "table",
         "extra": {"caption": "Table 1: numbers"}},
        {"doc_id": "d", "page_num": 3, "text": "12 40 Sales", "source_type": "pymupdf_figure", "modality": "image",
         "extra": {"image_path": "images/d/p003_fig0.jpg", "caption": "Figure 2: sales"}},
    ]
    chunks = chunk_document(blocks, {"max_chars": 500, "max_table_rows_per_chunk": 25})
    assert {c.page_num for c in chunks if c.modality == "text"} == {1, 2}
    tables = [c for c in chunks if c.modality == "table"]
    assert len(tables) == 3 and all("| a | b |" in t.content for t in tables)
    img = [c for c in chunks if c.modality == "image"][0]
    assert "Figure 2: sales" in img.content and img.image_path.endswith(".jpg")
    assert all(c.text.startswith("Document: Title of the doc") for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


# retrieval fusion
def test_rrf_prefers_items_found_by_both_retrievers():
    a = [{"chunk_id": "x"}, {"chunk_id": "y"}, {"chunk_id": "z"}]
    b = [{"chunk_id": "z"}, {"chunk_id": "w"}]
    fused = rrf_fuse([(a, 1.0), (b, 1.0)])
    assert fused[0]["chunk_id"] == "z"


# metrics
@pytest.mark.parametrize("pred,gold,fmt,exp", [
    ("Less well-off [2]", "Less well-off", "Str", 1.0),
    ("The answer is 72% [1]", "72", "Int", 1.0),
    ("0.72", "72%", "Float", 1.0),
    ("Not answerable", "Not answerable", "None", 1.0),
    ("It is 5 [1]", "Not answerable", "None", 0.0),
    ("Not answerable", "12", "Int", 0.0),
    ("['Apple', 'Pear'] are listed", "['Apple', 'Pear']", "List", 1.0),
    ("Apple only", "['Apple', 'Pear']", "List", 0.5),
])
def test_score_answer(pred, gold, fmt, exp):
    assert score_answer(pred, gold, fmt) == pytest.approx(exp)


def test_page_metrics():
    m = page_metrics([3, 3, 5, 9, 1], [5, 1])
    assert m["hit@1"] == 0 and m["hit@3"] == 1 and m["recall@5"] == 1.0
    assert m["mrr"] == pytest.approx(1 / 2)


def test_summarize_hallucination():
    rows = [
        {"answerable": True, "score": 1.0, "abstained": False},
        {"answerable": True, "score": 0.0, "abstained": False},    # wrong answer
        {"answerable": False, "score": 0.0, "abstained": False},   # answered an unanswerable question
        {"answerable": False, "score": 1.0, "abstained": True},
    ]
    s = summarize(rows)
    assert s["hallucination_rate"] == pytest.approx(0.5)
    assert s["hallucination_rate_unanswerable"] == pytest.approx(0.5)


# end to end on data/raw/check.pdf with hash models
@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF missing")
def test_end_to_end(tmp_path):
    from src.embeddings.models import ClipEmbedder, TextEmbedder
    from src.ingestion.pipeline import ingest_file
    from src.pipeline.rag import MultimodalRAG
    from src.utils.config import load_config
    from src.utils.metadata import save_jsonl
    from src.vectorstore.chroma_store import ChromaStore, build_index

    art = tmp_path / "artifacts"
    cfg = load_config(overrides={
        "paths": {"artifacts_dir": str(art), "processed_dir": str(tmp_path / "processed"),
                  "chroma_dir": str(art / "chroma")},
        "extraction": {"use_ocr": False},
        "models": {"reranker": None},
        "generation": {"backend": "extractive"},
    })
    blocks = ingest_file(SAMPLE_PDF, cfg, doc_id="check")
    assert any(b["modality"] == "image" for b in blocks)
    chunks = [c.to_dict() for c in chunk_document(blocks, cfg["chunking"])]
    save_jsonl(chunks, art / "chunks.jsonl")

    te, clip = TextEmbedder("hash:64"), ClipEmbedder("hash:32")
    np.save(art / "text_embeddings.npy", te.embed_documents([c["text"] for c in chunks]))
    imgs = [c for c in chunks if c["modality"] == "image"]
    np.save(art / "image_embeddings.npy", clip.embed_images([c["image_path"] for c in imgs]))
    (art / "image_chunk_ids.json").write_text(json.dumps([c["chunk_id"] for c in imgs]))
    (art / "manifest.json").write_text(json.dumps({"text_embedder": "hash:64", "clip": "hash:32"}))

    build_index(art, art / "chroma")
    rag = MultimodalRAG(cfg, ChromaStore(art / "chroma"),
                        json.loads((art / "manifest.json").read_text()), backend="extractive")
    out = rag.answer("Life Programmable Interface platform testing", doc_ids=["check"])
    assert out["contexts"] and out["contexts"][0]["doc_id"] == "check"
    assert out["answer"]
    nr = rag.answer("anything", mode="no_rag")
    assert nr["abstained"]

    # exact numpy search (Windows fallback) must return the same top hit as HNSW
    exact = ChromaStore(art / "chroma", exact_search=True)
    qv = te.embed_query("Life Programmable Interface platform testing")
    a = exact.query_text(qv, 5, ["check"])
    b = ChromaStore(art / "chroma").query_text(qv, 5, ["check"])
    assert a[0]["chunk_id"] == b[0]["chunk_id"]
