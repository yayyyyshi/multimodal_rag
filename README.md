# Retrieval-Augmented Multimodal Reasoning over Images, Text, Tables and Reports

A multimodal RAG system built from scratch. It answers questions about PDFs, images and
spreadsheets by **retrieving evidence** (text passages, tables, charts and figures) and letting an
**open-source LLM** (Qwen) reason over it. Every answer **cites its sources**, and the model says
*"Not answerable"* when the evidence isn't there.

The retrieval pipeline is written by hand: no LangChain, no LlamaIndex, no paid APIs.

---

## Sem VII objectives and where each one is met

| Objective | Implementation | Measured by |
|---|---|---|
| **Multimodal retrieval system** (images, documents, tables, text) | `src/extraction` pulls text, tables (Markdown) and figures (embedded images + vector charts, cropped). OCR covers scanned pages and chart labels. Qwen2.5-VL describes each figure (chart type, labels, values); BLIP is kept as a fallback. Sentence-Transformers embeds text, CLIP embeds images. Everything goes into two ChromaDB collections that are fused at query time. | Retrieval split by evidence type (Text / Layout / Table / Chart / Figure), multimodal vs text-only |
| **Contextual retrieval** (only the most relevant knowledge, top-k) | Contextual chunk headers ("Document / Page / type"). Dense text search plus CLIP image search, merged with Reciprocal Rank Fusion. Optional cross-encoder rerank. Top-k passed to the LLM. | Hit@k, Recall@k, MRR on the gold evidence pages |
| **Reduce hallucinations** (ground LLM outputs) | Grounded prompt with a mandatory `[n]` citation for each fact. A weak-evidence gate. An explicit abstention answer ("Not answerable"). | Accuracy and hallucination rate: **LLM alone vs text RAG vs multimodal RAG** on 54 unanswerable and 148 answerable questions |

## Architecture

```
            ┌──────────────── Kaggle GPU (heavy, run once) ────────────────┐
PDF/IMG/CSV │ PyMuPDF text+tables+figures ─► PaddleOCR (scanned pages,      │
            │ chart labels) ─► cleaning ─► chunking (+context header)       │
            │ ─► Qwen2.5-VL figure descriptions ─► bge-small + CLIP         │
            │ ─► artifacts.zip  (chunks.jsonl, *.npy, images/, manifest)    │
            └───────────────────────────────┬───────────────────────────────┘
                                            │ download
            ┌──────────────── Your laptop (CPU) ────────▼───────────────────┐
question ──►│ bge query emb ─► ChromaDB mm_text ─┐                           │
            │ CLIP text emb ─► ChromaDB mm_image ─┴► RRF ─► rerank ─► top-k  │
            │ ─► grounded prompt (+ images for VLM) ─► Qwen ─► answer [n]    │
            └────────────────────────────────────────────────────────────────┘
```

Why the split? Building embeddings means running CLIP, the figure describer and OCR over thousands of chunks and
images, which needs a GPU. At query time only **one** question has to be embedded, and bge-small and
CLIP-B/32 handle that on CPU in milliseconds. **The query must use the same models as the
artifacts.** `manifest.json` records which models were used, and the code always loads those.

## Dataset: MMLongBench-Doc (NeurIPS 2024 D&B, Apache-2.0)

The benchmark has 135 real long PDFs (research reports, papers, brochures, financial reports,
manuals, slide decks) and 1,082 questions. Each question is labelled with its **evidence pages** and
**evidence type** (Pure-text, Layout, Table, Chart, Figure), and 20% of them are **"Not answerable"**.
We use a stratified **20-document subset** (≤60 pages each) with 202 questions, 54 of them
unanswerable. See `data/eval/`.

## Project layout

```
configs/config.yaml          all paths, model names, parameters (one place)
src/
  utils/metadata.py          shared schemas: ExtractedBlock, Chunk
  extraction/                pymupdf_extractor.py (text/tables/figures), ocr_extractor.py (PaddleOCR)
  preprocessing/cleaner.py   noise, running headers, safe OCR fixes, confidence flags
  ingestion/pipeline.py      routes PDF / image / CSV-XLSX / TXT -> blocks
  chunking/chunker.py        page-bounded chunks, table splitting, figure descriptions
  embeddings/models.py       TextEmbedder, ClipEmbedder, VLCaptioner (Qwen2.5-VL), Captioner (BLIP), Reranker
  vectorstore/chroma_store.py
  retrieval/retriever.py     multimodal fusion retriever
  generation/                prompts + backends (transformers, transformers_vl, ollama, extractive)
  pipeline/rag.py            MultimodalRAG.answer(...)
  evaluation/                dataset loader + metrics
scripts/                     prepare_dataset -> ingest -> build_embeddings -> build_index -> ask / evaluate
notebooks/                   01 Kaggle: build artifacts · 02 Kaggle: LLM + evaluation
app/streamlit_app.py         demo UI
tests/                       pytest (no GPU / no downloads)
```

## How to run

### A. Kaggle (heavy work, about 30–40 min)
1. kaggle.com → *New Notebook* → *File → Import notebook* → `notebooks/01_kaggle_build_artifacts.ipynb`
2. Settings: **GPU T4**, **Internet ON** (Internet needs a phone-verified Kaggle account).
3. Set `BRANCH` in the first cell, then *Run all*. Download `multimodal_rag/artifacts.zip` from **Output**.
4. (Optional, for the report numbers) import `02_kaggle_rag_evaluation.ipynb`, *Add Input* → notebook 01's
   output, *Run all*. You get `results/kaggle_full/summary.md` and `comparison.png`.

### B. Laptop (Windows, VS Code, CPU)
```bash
python -m venv venv
venv\Scripts\activate                       # Linux/Mac: source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python scripts/build_index.py --zip artifacts.zip        # unzip + load into ChromaDB (seconds)
python scripts/ask.py "How do 5% of Latinos see upward mobility for their children?"
streamlit run app/streamlit_app.py
python scripts/evaluate.py --retrieval-only              # retrieval metrics on CPU
pytest -q
```

**LLM on a CPU laptop:** install [Ollama](https://ollama.com), run `ollama pull qwen2.5vl:3b`, then pick the
`ollama` backend (it's the fastest on CPU and it can see the images). With the `transformers` backend,
Qwen2.5-1.5B-Instruct runs through Hugging Face. It works, but it's slower.

### Adding your own documents
Put PDFs, images, CSV or XLSX files in `data/raw/`. Then run `python scripts/ingest.py --inputs data/raw` on
Kaggle or locally with `--no-ocr`, followed by `build_embeddings.py` and `build_index.py`.

## Configuration notes
* Change models in `configs/config.yaml`. If you change `text_embedder` or `clip`, **rebuild the artifacts**.
* `generation.abstain_policy: hard` refuses without calling the LLM when evidence is weak. That's stricter,
  so there are fewer hallucinations but more over-refusals.
* `hash:<dim>` model names give a dependency-free fake embedder. It's only for tests and smoke runs.

## Status
See [`docs/PROGRESS.md`](docs/PROGRESS.md).
