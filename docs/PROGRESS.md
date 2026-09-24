# Progress vs PERT chart (Minor Project, 12 weeks)

| Week | Milestone | Status | Where |
|---|---|---|---|
| 1–2 | Literature review & problem finalisation | ✅ done | synopsis slides |
| 3–4 | Data collection & preprocessing (OCR, cleaning) | ✅ code done | `scripts/prepare_dataset.py`, `src/extraction`, `src/preprocessing`, `src/ingestion` |
| 5–6 | Embedding generation (CLIP/BLIP + Sentence-Transformers) | ✅ code done · ⏳ run on Kaggle | `src/embeddings`, `scripts/build_embeddings.py`, notebook 01 |
| 7 | Vector store (ChromaDB) | ✅ done | `src/vectorstore`, `scripts/build_index.py` |
| 8 | Retrieval pipeline integration | ✅ done | `src/retrieval` |
| 9–10 | LLM integration & multimodal reasoning | ✅ code done · ⏳ run on Kaggle | `src/generation`, `src/pipeline`, notebook 02 |
| 11 | Testing & evaluation | ✅ harness + unit tests · ⏳ real numbers from Kaggle | `scripts/evaluate.py`, `tests/` |
| 12 | Documentation & clean repository | 🟡 README done; report chapters pending | `README.md` |

## History
* **Aug 27** (Yashika): PaddleOCR extraction + cleaning.
* **Sep 6** (Jyoti): version fixes, PyMuPDF native text extractor.
* **Sep 24**: full pipeline.
  * Extractor gained tables, figures and charts, and now routes only scanned pages to OCR (it used to OCR every page).
  * Cleaner no longer corrupts valid text. The old `l→I` and `0→O` rules were removed.
  * PyMuPDF extractor no longer emits `<image: ...>` blocks as text.
  * Added chunking, embeddings, ChromaDB, retrieval, the Qwen generator, evaluation, Kaggle notebooks and the Streamlit app.

## Next (Sem VIII extensions, already partly prepared)
* Explainability: citations → highlight bbox on the page image (`bbox` is stored for every chunk).
* Larger collections: run notebook 01 with `dataset.n_docs: 135`.
* Latency/accuracy benchmark vs baseline RAG: `evaluate.py` already reports latency per mode.
