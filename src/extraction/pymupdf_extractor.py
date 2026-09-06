"""
PyMuPDF-based text extraction for machine-readable PDF pages.

Handles pages that already contain a text layer (native PDFs, exported
reports, digital documents) — this is the fast path that doesn't need
OCR. Scanned / image-only pages return no text here and should be routed
to Yashika's PaddleOCR pipeline (src/extraction/ocr_extractor.py) instead.

This is a starter/stub implementation: the core extraction works end to
end and produces output in the shared ExtractedBlock schema, but it's
intentionally simple (block-level text only, no table structure yet) so
Jyoti can extend it — e.g. add table detection, better block merging,
or font/heading metadata — without breaking the schema contract.

Usage:
    python src/extraction/pymupdf_extractor.py --pdf data/raw/sample.pdf --doc-id sample
"""

import argparse
import os
import sys

import pymupdf as fitz  # PyMuPDF

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.metadata import ExtractedBlock, save_blocks

# A page is considered "scanned" (no usable text layer) if the total
# extracted text is shorter than this many characters.
MIN_TEXT_CHARS_FOR_NATIVE_PAGE = 10


def extract_text_blocks_from_page(page: "fitz.Page") -> list[dict]:
    """
    Extract text blocks from a single page using PyMuPDF's block-level
    text extraction. Each block already comes with a bounding box.

    Returns: [{"text": ..., "bbox": [x0, y0, x1, y1]}, ...]
    """
    raw_blocks = page.get_text("blocks")  # each: (x0, y0, x1, y1, text, block_no, block_type)
    blocks = []

    for b in raw_blocks:
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
        text = text.strip()
        if not text:
            continue
        blocks.append({
            "text": text,
            "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
        })

    return blocks


def extract_pymupdf_blocks(pdf_path: str, doc_id: str) -> tuple[list[ExtractedBlock], list[int]]:
    """
    Full pipeline: PDF -> per-page native text blocks -> ExtractedBlock list.

    Also returns a list of 1-indexed page numbers that appear to have no
    usable text layer (likely scanned images), so those specific pages
    can be handed off to the OCR pipeline instead of re-running OCR on
    every page.
    """
    doc = fitz.open(pdf_path)
    all_blocks = []
    scanned_page_candidates = []

    for page_num, page in enumerate(doc, start=1):
        print(f"[PyMuPDF] Processing page {page_num}/{len(doc)} ...")
        page_blocks = extract_text_blocks_from_page(page)

        total_chars = sum(len(b["text"]) for b in page_blocks)
        if total_chars < MIN_TEXT_CHARS_FOR_NATIVE_PAGE:
            scanned_page_candidates.append(page_num)
            continue

        for b in page_blocks:
            all_blocks.append(ExtractedBlock(
                doc_id=doc_id,
                page_num=page_num,
                text=b["text"],
                source_type="pymupdf_text",
                confidence=None,   # not applicable for native text extraction
                bbox=b["bbox"],
            ))

    doc.close()
    return all_blocks, scanned_page_candidates


def main():
    parser = argparse.ArgumentParser(description="Run PyMuPDF native-text extraction on a PDF")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--doc-id", required=True, help="Identifier for this document")
    parser.add_argument("--out", default=None, help="Output JSON path (default: data/processed/<doc_id>_pymupdf.json)")
    args = parser.parse_args()

    out_path = args.out or f"data/processed/{args.doc_id}_pymupdf.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    blocks, scanned_pages = extract_pymupdf_blocks(args.pdf, args.doc_id)
    save_blocks(blocks, out_path)

    print(f"[DONE] Extracted {len(blocks)} native text blocks -> {out_path}")
    if scanned_pages:
        print(
            f"[INFO] {len(scanned_pages)} page(s) had little/no text layer "
            f"(likely scanned): {scanned_pages}"
        )
        print(
            "[INFO] Route these pages to ocr_extractor.py for OCR-based "
            "extraction instead of re-running OCR on the whole document."
        )


if __name__ == "__main__":
    main()
