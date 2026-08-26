"""
OCR-based text extraction using PaddleOCR.

Handles scanned pages / image-based content that PyMuPDF cannot
extract text from directly (Jyoti's extractor handles the
machine-readable pages; this file handles the rest).

Usage:
    python src/extraction/ocr_extractor.py --pdf data/raw/sample.pdf --doc-id sample
"""

import argparse
import os
import sys

from paddleocr import PaddleOCR
import pymupdf as fitz  # PyMuPDF, used only to rasterize pages to images

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.metadata import ExtractedBlock, save_blocks


def pdf_pages_to_images(pdf_path: str, out_dir: str, dpi: int = 200) -> list[str]:
    """
    Convert every page of a PDF into a PNG image.
    Returns list of image file paths, one per page (1-indexed order).
    """
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []

    zoom = dpi / 72  # PDF default is 72 dpi
    matrix = fitz.Matrix(zoom, zoom)

    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=matrix)
        img_path = os.path.join(out_dir, f"page_{i}.png")
        pix.save(img_path)
        image_paths.append(img_path)

    doc.close()
    return image_paths


def run_ocr_on_image(ocr_engine: PaddleOCR, image_path: str) -> list[dict]:
    """
    Run PaddleOCR on a single image and return raw line-level results:
    [{"text": ..., "confidence": ..., "bbox": [...]}, ...]

    Uses the PaddleOCR 3.x API: `.predict()` instead of the old `.ocr()`.
    Each result object behaves like a dict with keys: rec_texts, rec_scores,
    rec_boxes (already [x0, y0, x1, y1] per line — no corner-point math needed).
    """
    results = ocr_engine.predict(image_path)
    lines = []

    for res in results:
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        boxes = res.get("rec_boxes", [])  # each box: [x0, y0, x1, y1]

        for text, score, box in zip(texts, scores, boxes):
            lines.append({
                "text": text,
                "confidence": float(score),
                "bbox": [float(v) for v in box],
            })

    return lines


def group_lines_into_blocks(lines: list[dict], y_threshold: float = 15.0) -> list[dict]:
    """
    Merge OCR lines that are close together vertically into paragraph-like
    blocks, so we don't store one JSON entry per single line of text.
    """
    if not lines:
        return []

    # sort top-to-bottom
    sorted_lines = sorted(lines, key=lambda l: l["bbox"][1])

    blocks = []
    current = [sorted_lines[0]]

    for line in sorted_lines[1:]:
        prev_bottom = current[-1]["bbox"][3]
        this_top = line["bbox"][1]
        if this_top - prev_bottom <= y_threshold:
            current.append(line)
        else:
            blocks.append(current)
            current = [line]
    blocks.append(current)

    merged = []
    for block_lines in blocks:
        text = " ".join(l["text"] for l in block_lines)
        avg_conf = sum(l["confidence"] for l in block_lines) / len(block_lines)
        xs0 = min(l["bbox"][0] for l in block_lines)
        ys0 = min(l["bbox"][1] for l in block_lines)
        xs1 = max(l["bbox"][2] for l in block_lines)
        ys1 = max(l["bbox"][3] for l in block_lines)
        merged.append({
            "text": text,
            "confidence": avg_conf,
            "bbox": [xs0, ys0, xs1, ys1],
        })

    return merged


def extract_ocr_blocks(pdf_path: str, doc_id: str, tmp_image_dir: str = "data/processed/_tmp_pages") -> list[ExtractedBlock]:
    """
    Full pipeline: PDF -> page images -> OCR -> grouped blocks -> ExtractedBlock list.
    """
    # PaddleOCR 3.x: `use_angle_cls` and `show_log` were removed/renamed.
    ocr_engine = PaddleOCR(use_textline_orientation=True, lang="en")

    image_paths = pdf_pages_to_images(pdf_path, tmp_image_dir)
    all_blocks = []

    for page_num, img_path in enumerate(image_paths, start=1):
        print(f"[OCR] Processing page {page_num}/{len(image_paths)} ...")
        raw_lines = run_ocr_on_image(ocr_engine, img_path)
        grouped = group_lines_into_blocks(raw_lines)

        for g in grouped:
            all_blocks.append(ExtractedBlock(
                doc_id=doc_id,
                page_num=page_num,
                text=g["text"],
                source_type="paddleocr",
                confidence=round(g["confidence"], 4),
                bbox=[round(v, 2) for v in g["bbox"]],
            ))

    return all_blocks


def main():
    parser = argparse.ArgumentParser(description="Run PaddleOCR extraction on a PDF")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--doc-id", required=True, help="Identifier for this document")
    parser.add_argument("--out", default=None, help="Output JSON path (default: data/processed/<doc_id>_ocr.json)")
    args = parser.parse_args()

    out_path = args.out or f"data/processed/{args.doc_id}_ocr.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    blocks = extract_ocr_blocks(args.pdf, args.doc_id)
    save_blocks(blocks, out_path)

    print(f"[DONE] Extracted {len(blocks)} OCR blocks -> {out_path}")


if __name__ == "__main__":
    main()
