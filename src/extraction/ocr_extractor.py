
"""Tesseract OCR extraction for scanned PDF pages and images."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pymupdf as fitz
import pytesseract
from PIL import Image
from pytesseract import Output

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.metadata import TEXT, ExtractedBlock, save_blocks


# Update this path if Tesseract is installed elsewhere.
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if Path(TESSERACT_EXE).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE


def get_ocr_engine(lang: str = "eng") -> str:
    """Verify Tesseract is available and return the OCR language."""
    pytesseract.get_tesseract_version()
    return lang


def pdf_pages_to_images(
    pdf_path: str | Path,
    out_dir: str | Path,
    dpi: int = 200,
    pages: list[int] | None = None,
) -> dict[int, str]:
    """Render selected 1-indexed PDF pages to PNG files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = {}
    doc = fitz.open(pdf_path)

    try:
        selected = (
            pages
            if pages is not None
            else range(1, len(doc) + 1)
        )

        for page_num in selected:
            if not 1 <= page_num <= len(doc):
                continue

            page = doc[page_num - 1]
            image_path = out_dir / f"page_{page_num}.png"

            page.get_pixmap(dpi=dpi, alpha=False).save(str(image_path))
            rendered[page_num] = str(image_path)
    finally:
        doc.close()

    return rendered


def run_ocr_on_image(
    image_path: str | Path,
    engine: str | None = None,
) -> list[dict]:
    """
    Extract text lines from an image.

    Returns dictionaries containing:
      text: recognized text
      confidence: normalized score from 0 to 1
      bbox: [x0, y0, x1, y1] in image pixels
    """
    lang = engine or "eng"

    with Image.open(image_path) as image:
        data = pytesseract.image_to_data(
            image.convert("RGB"),
            lang=lang,
            config="--psm 6",
            output_type=Output.DICT,
        )

    grouped = defaultdict(list)

    for i, word in enumerate(data["text"]):
        word = word.strip()

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            continue

        if not word or confidence < 0:
            continue

        key = (
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i],
        )

        left = int(data["left"][i])
        top = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])

        grouped[key].append({
            "text": word,
            "confidence": confidence / 100.0,
            "bbox": [
                left,
                top,
                left + width,
                top + height,
            ],
        })

    lines = []

    for words in grouped.values():
        words.sort(key=lambda item: item["bbox"][0])

        boxes = [word["bbox"] for word in words]

        lines.append({
            "text": " ".join(word["text"] for word in words),
            "confidence": sum(
                word["confidence"] for word in words
            ) / len(words),
            "bbox": [
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            ],
        })

    lines.sort(key=lambda line: (line["bbox"][1], line["bbox"][0]))
    return lines


def group_lines_into_blocks(
    lines: list[dict],
    y_threshold: float = 15.0,
) -> list[dict]:
    """Combine vertically adjacent OCR lines into text blocks."""
    if not lines:
        return []

    lines = sorted(
        lines,
        key=lambda line: (line["bbox"][1], line["bbox"][0]),
    )

    blocks = []
    current = [lines[0]]

    for line in lines[1:]:
        previous = current[-1]
        vertical_gap = line["bbox"][1] - previous["bbox"][3]

        if vertical_gap <= y_threshold:
            current.append(line)
        else:
            blocks.append(_make_block(current))
            current = [line]

    blocks.append(_make_block(current))
    return blocks


def _make_block(lines: list[dict]) -> dict:
    """Combine OCR lines and their bounding boxes."""
    boxes = [line["bbox"] for line in lines]

    return {
        "text": "\n".join(line["text"] for line in lines),
        "confidence": sum(
            line["confidence"] for line in lines
        ) / len(lines),
        "bbox": [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ],
    }


def ocr_image_text(
    image_path: str | Path,
    min_conf: float = 0.5,
) -> str:
    """Return recognized lines meeting the confidence threshold."""
    lines = run_ocr_on_image(image_path)

    return "\n".join(
        line["text"]
        for line in lines
        if line["confidence"] >= min_conf
    )


def extract_ocr_blocks(
    pdf_path: str | Path,
    doc_id: str,
    pages: list[int] | None = None,
    tmp_image_dir: str | Path = "data/processed/_tmp_pages",
    dpi: int = 200,
) -> list[ExtractedBlock]:
    """
    OCR selected PDF pages and return ExtractedBlock objects.

    Bounding boxes are converted from image pixels to PDF points.
    """
    lang = get_ocr_engine()

    rendered_pages = pdf_pages_to_images(
        pdf_path=pdf_path,
        out_dir=tmp_image_dir,
        dpi=dpi,
        pages=pages,
    )

    blocks = []

    for page_num, image_path in rendered_pages.items():
        lines = run_ocr_on_image(image_path, engine=lang)
        text_blocks = group_lines_into_blocks(lines)

        scale = 72.0 / dpi

        for block in text_blocks:
            if not block["text"].strip():
                continue

            x0, y0, x1, y1 = block["bbox"]

            pdf_bbox = [
                round(x0 * scale, 2),
                round(y0 * scale, 2),
                round(x1 * scale, 2),
                round(y1 * scale, 2),
            ]

            blocks.append(
                ExtractedBlock(
                    doc_id=doc_id,
                    page_num=page_num,
                    text=block["text"],
                    source_type="tesseract",
                    confidence=block["confidence"],
                    bbox=pdf_bbox,
                    modality=TEXT,
                )
            )

        print(
            f"[Tesseract] Page {page_num}: "
            f"{len(text_blocks)} text blocks"
        )

    return blocks


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from PDF pages using Tesseract OCR"
    )
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--pages", nargs="*", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path",
    )
    parser.add_argument(
        "--tmp-image-dir",
        default="data/processed/_tmp_pages",
    )

    args = parser.parse_args()

    blocks = extract_ocr_blocks(
        pdf_path=args.pdf,
        doc_id=args.doc_id,
        pages=args.pages,
        tmp_image_dir=args.tmp_image_dir,
        dpi=args.dpi,
    )

    out_path = args.out or (
        f"data/processed/{args.doc_id}/ocr.json"
    )
    save_blocks(blocks, out_path)

    print(f"[DONE] Extracted {len(blocks)} OCR blocks -> {out_path}")


if __name__ == "__main__":
    main()
