# Multimodal RAG — Yashika's Branch (OCR + Preprocessing)

This covers **your portion** of the project: OCR-based extraction (PaddleOCR)
and cleaning/structuring of that extracted content. Jyoti's PyMuPDF
extraction lives in a parallel file (`src/extraction/pymupdf_extractor.py`)
that she'll add — both of you write to the **same shared schema** in
`src/utils/metadata.py` so the outputs merge cleanly later.

## 1. First-time setup (do this once)

### Option A — Test in Google Colab first (recommended for first run)
1. Open https://colab.research.google.com, new notebook.
2. Runtime → Change runtime type → GPU (optional but faster).
3. Run:
   ```
   !pip install paddlepaddle paddleocr PyMuPDF opencv-python pillow
   ```
4. Upload a sample scanned PDF and test that PaddleOCR runs without errors
   before touching the real repo. This just de-risks the install.

### Option B — Local setup in VSCode (for actual project work)
```bash
# clone the repo (after it's created on GitHub)
git clone <repo-url>
cd multimodal-rag

# create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```
Open the folder in VSCode, install the Python extension if prompted.

## 2. Running your part

```bash
# 1. Run OCR extraction on a sample PDF
python src/extraction/ocr_extractor.py --pdf data/raw/sample.pdf --doc-id sample

# 2. Clean the raw OCR output
python src/preprocessing/cleaner.py --in data/processed/sample_ocr.json --out data/processed/sample_clean.json
```

Output: `data/processed/sample_clean.json` — a list of text blocks with
page number, confidence score, bounding box, and a `flagged_low_confidence`
field for anything that needs manual review.

## 3. Git / GitHub workflow (two people, PR-based)

1. **Create the repo once** (either person), add the other as collaborator.
2. `main` stays stable — never commit directly to it.
3. Each person works on their own branch:
   ```bash
   git checkout -b feature/ocr-extraction     # you
   git checkout -b feature/pymupdf-extraction # Jyoti
   ```
4. Commit in small chunks with clear messages:
   ```bash
   git add src/extraction/ocr_extractor.py
   git commit -m "ocr: add PaddleOCR wrapper and page rasterization"
   ```
5. Push and open a PR into `main`:
   ```bash
   git push origin feature/ocr-extraction
   ```
   Then open the PR on GitHub, tag Jyoti as reviewer.
6. Review each other's PRs before merging — mainly check that the output
   still matches the shared schema in `metadata.py`.
7. Pull the latest `main` before starting new work each session:
   ```bash
   git checkout main
   git pull
   ```

## 4. Why separate files avoid merge conflicts

You and Jyoti each own separate files (`ocr_extractor.py` vs
`pymupdf_extractor.py`), so your PRs touch different files and rarely
conflict. The only shared file is `src/utils/metadata.py` — agree on any
changes to it together before editing.

## 5. Next steps after this stage is validated
- Document chunking
- Multimodal embedding generation
- Vector database integration (Qdrant/ChromaDB)
- Semantic retrieval
- Reasoning + response generation
