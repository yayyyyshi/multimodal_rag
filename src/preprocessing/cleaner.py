"""Cleans extracted blocks from both OCR and native text.

Normalizes whitespace, drops page numbers and divider lines, removes headers
and footers repeated across pages, and flags OCR blocks below
CONFIDENCE_THRESHOLD (they are kept, only marked).

OCR character fixes are applied only between digits. The earlier rules
(l -> I, 0 -> O) also changed valid text such as "0x1F" and "5G".

Usage:
    python src/preprocessing/cleaner.py --in data/processed/sample_ocr.json --out data/processed/sample_clean.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.metadata import load_blocks

CONFIDENCE_THRESHOLD = 0.60

# "1O5" -> "105", "2l3" -> "213"
OCR_FIXES = [
    (re.compile(r"(?<=\d)[Oo](?=\d)"), "0"),
    (re.compile(r"(?<=\d)[lI](?=\d)"), "1"),
    (re.compile(r"\s+([,.;:])(?=\s|$)"), r"\1"),  # "word ," -> "word,"
]

NOISE_PATTERNS = [
    re.compile(r"^\s*\d{1,4}\s*$"),                             # lone page number
    re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.I),      # "Page 3 of 10"
    re.compile(r"^\s*[-_=.·•]{3,}\s*$"),                        # divider lines
]


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"-\s*\n\s*(?=[a-z])", "", text)
    return re.sub(r"\s+", " ", text).strip()


def apply_ocr_fixes(text: str) -> str:
    for pattern, repl in OCR_FIXES:
        text = pattern.sub(repl, text)
    return text


def is_noise(text: str) -> bool:
    t = text.strip()
    return not t or any(p.match(t) for p in NOISE_PATTERNS)


def _header_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def find_repeated_headers(blocks: list[dict], min_pages: int = 3, min_ratio: float = 0.4) -> set[str]:
    """Short lines found on at least min_ratio of the pages, e.g. 'PEW RESEARCH CENTER'."""
    pages = {b["page_num"] for b in blocks}
    if len(pages) < min_pages:
        return set()
    seen = Counter()
    for key in {(b["page_num"], _header_key(b["text"]))
                for b in blocks if b.get("modality", "text") == "text" and len(b["text"]) < 80}:
        seen[key[1]] += 1
    return {t for t, n in seen.items() if n >= max(min_pages, min_ratio * len(pages))}


def clean_block(block: dict, is_ocr: bool | None = None) -> dict | None:
    """Returns None if the block is noise."""
    text = normalize_whitespace(block.get("text", ""))
    if is_ocr is None:
        is_ocr = block.get("source_type") == "paddleocr"
    if is_ocr:
        text = apply_ocr_fixes(text)
    modality = block.get("modality", "text")
    if modality == "text" and is_noise(text):
        return None

    confidence = block.get("confidence")
    block["text"] = text if modality != "table" else block["text"]  # keep Markdown line breaks
    block["extra"] = dict(block.get("extra") or {})
    block["extra"]["flagged_low_confidence"] = confidence is not None and confidence < CONFIDENCE_THRESHOLD
    return block


def clean_blocks(raw_blocks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (clean_blocks, flagged_blocks)."""
    headers = find_repeated_headers(raw_blocks)
    cleaned, flagged = [], []
    for b in raw_blocks:
        if b.get("modality", "text") == "text" and \
                _header_key(b.get("text", "")) in headers:
            continue
        result = clean_block(dict(b))
        if result is None:
            continue
        cleaned.append(result)
        if result["extra"]["flagged_low_confidence"]:
            flagged.append(result)
    return cleaned, flagged


def main():
    ap = argparse.ArgumentParser(description="Clean extracted blocks")
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    args = ap.parse_args()

    raw = load_blocks(args.in_path)
    cleaned, flagged = clean_blocks(raw)
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Cleaned {len(cleaned)} blocks (from {len(raw)} raw) -> {args.out_path}")
    print(f"[INFO] {len(flagged)} blocks flagged as low-confidence for manual review.")


if __name__ == "__main__":
    main()
