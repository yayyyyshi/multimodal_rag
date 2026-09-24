"""PaddleOCR 3.x text extraction.

Used for scanned PDF pages (only the ones the PyMuPDF extractor flags),
standalone image files, and chart crops whose labels are drawn as paths
instead of text. PaddleOCR is imported only when needed, so the rest of the
project works without it installed.

Usage:
    python src/extraction/ocr_extractor.py --pdf data/raw/sample.pdf --doc-id sample
    python src/extraction/ocr_extractor.py --pdf data/raw/sample.pdf --doc-id sample --pages 1 4 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pymupdf as fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.metadata import TEXT, ExtractedBlock, save_blocks

_ENGINE = None


def get_ocr_engine(lang: str = "en"):
    """Loaded once per process because model loading takes several seconds."""
    global _ENGINE
    if _ENGINE is None:
        from paddleocr import PaddleOCR
        # document unwarping and orientation models are slow and not needed for PDF renders
        base = dict(lang=lang, use_textline_orientation=True,
                    use_doc_orientation_classify=False, use_doc_unwarping=False)
        try:  # mobile detector is several times faster on CPU
            _ENGINE = PaddleOCR(text_detection_model_name="PP-OCRv5_mobile_det", **base)
        except Exception:
            _ENGINE = PaddleOCR(**base)
    return _ENGINE


def pdf_pages_to_images(pdf_path, out_dir, dpi: int = 200, pages: list[int] | None = None) -> dict[int, str]:
    """Returns {page_num: png_path} for the requested 1-indexed pages (all if None)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    result = {}
    for i, page in enumerate(doc, start=1):
        if pages and i not in pages:
            continue
        img_path = out_dir / f"page_{i}.png"
        page.get_pixmap(dpi=dpi).save(str(img_path))
        result[i] = str(img_path)
    doc.close()
    return result


def run_ocr_on_image(image_path: str, engine=None) -> list[dict]:
    """Returns one dict per detected line: text, confidence, bbox in pixels."""
    engine = engine or get_ocr_engine()
    lines = []
    for res in engine.predict(str(image_path)):
        for text, score, box in zip(res.get("rec_texts", []), res.get("rec_scores", []), res.get("rec_boxes", [])):
            lines.append({"text": text, "confidence": float(score), "bbox": [float(v) for v in box]})
    return lines


def group_lines_into_blocks(lines: list[dict], y_threshold: float = 15.0) -> list[dict]:
    """Merges lines whose vertical gap is at most y_threshold pixels into one block."""
    if not lines:
        return []
    sorted_lines = sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0]))
    groups, current = [], [sorted_lines[0]]
    for line in sorted_lines[1:]:
        if line["bbox"][1] - current[-1]["bbox"][3] <= y_threshold:
            current.append(line)
        else:
            groups.append(current)
            current = [line]
    groups.append(current)

    merged = []
    for g in groups:
        merged.append({
            "text": " ".join(l["text"] for l in g),
            "confidence": sum(l["confidence"] for l in g) / len(g),
            "bbox": [min(l["bbox"][0] for l in g), min(l["bbox"][1] for l in g),
                     max(l["bbox"][2] for l in g), max(l["bbox"][3] for l in g)],
        })
    return merged


def ocr_image_text(image_path: str, min_conf: float = 0.5) -> str:
    """All lines above min_conf joined into one string. Returns '' on failure."""
    try:
        lines = run_ocr_on_image(image_path)
    except Exception as e:
        print(f"[OCR] failed on {image_path}: {e}")
        return ""
    return " ".join(l["text"] for l in lines if l["confidence"] >= min_conf)


def extract_ocr_blocks(pdf_path, doc_id: str, pages: list[int] | None = None,
                       tmp_image_dir="data/processed/_tmp_pages", dpi: int = 200) -> list[ExtractedBlock]:
    """Renders the selected pages, runs OCR and returns one block per text group."""
    engine = get_ocr_engine()
    image_paths = pdf_pages_to_images(pdf_path, Path(tmp_image_dir) / doc_id, dpi=dpi, pages=pages)
    scale = 72.0 / dpi  # pixel boxes to PDF points, same units as PyMuPDF
    blocks = []
    for page_num, img_path in sorted(image_paths.items()):
        print(f"[OCR] {doc_id}: page {page_num}")
        for g in group_lines_into_blocks(run_ocr_on_image(img_path, engine)):
            blocks.append(ExtractedBlock(
                doc_id=doc_id, page_num=page_num, text=g["text"], source_type="paddleocr",
                confidence=round(g["confidence"], 4),
                bbox=[round(v * scale, 2) for v in g["bbox"]], modality=TEXT,
            ))
    return blocks


def main():
    ap = argparse.ArgumentParser(description="Run PaddleOCR extraction on a PDF")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--pages", type=int, nargs="*", default=None, help="1-indexed pages (default: all)")
    ap.add_argument("--out", default=None, help="default: data/processed/<doc_id>_ocr.json")
    args = ap.parse_args()

    out_path = args.out or f"data/processed/{args.doc_id}_ocr.json"
    blocks = extract_ocr_blocks(args.pdf, args.doc_id, pages=args.pages)
    save_blocks(blocks, out_path)
    print(f"[DONE] Extracted {len(blocks)} OCR blocks -> {out_path}")


if __name__ == "__main__":
    main()
