"""
Cleaning and structuring for OCR-extracted content.

Takes raw OCR output (from ocr_extractor.py) and:
  - removes noise (headers/footers, page numbers, junk symbols)
  - normalizes whitespace
  - flags/filters low-confidence detections
  - fixes a few common OCR character mistakes

Usage:
    python src/preprocessing/cleaner.py --in data/processed/sample_ocr.json --out data/processed/sample_clean.json
"""

import argparse
import json
import re
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.metadata import load_blocks, save_blocks, ExtractedBlock

# Minimum OCR confidence to keep a block without flagging it
CONFIDENCE_THRESHOLD = 0.60

# Common OCR misreads worth auto-correcting (extend this as you find more)
OCR_FIXES = {
    r"\bl\b": "I",       # lowercase L misread as capital I in isolation
    r"\b0(?=[A-Za-z])": "O",  # zero misread as letter O when touching letters
    r"\s+,": ",",
    r"\s+\.": ".",
}

# Patterns that usually indicate headers/footers/page numbers, not real content
NOISE_PATTERNS = [
    r"^\s*\d+\s*$",                  # a lone number (likely a page number)
    r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$",  # "Page 3 of 10"
    r"^\s*[-_=]{3,}\s*$",            # divider lines like "----"
]


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def apply_ocr_fixes(text: str) -> str:
    for pattern, replacement in OCR_FIXES.items():
        text = re.sub(pattern, replacement, text)
    return text


def is_noise(text: str) -> bool:
    stripped = text.strip().lower()
    if len(stripped) == 0:
        return True
    for pattern in NOISE_PATTERNS:
        if re.match(pattern, stripped, flags=re.IGNORECASE):
            return True
    return False


def clean_block(block: dict) -> dict | None:
    """
    Clean a single block dict. Returns None if the block should be dropped
    (pure noise), otherwise returns the cleaned block with a 'flagged' field
    added for low-confidence content.
    """
    text = block.get("text", "")
    text = normalize_whitespace(text)
    text = apply_ocr_fixes(text)

    if is_noise(text):
        return None

    confidence = block.get("confidence")
    flagged = confidence is not None and confidence < CONFIDENCE_THRESHOLD

    block["text"] = text
    block["extra"] = block.get("extra", {})
    block["extra"]["flagged_low_confidence"] = flagged

    return block


def clean_blocks(raw_blocks: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Returns (clean_blocks, flagged_blocks) — flagged_blocks are kept but
    marked for manual review (e.g. blurry scans).
    """
    cleaned = []
    flagged = []

    for b in raw_blocks:
        result = clean_block(dict(b))  # copy to avoid mutating input
        if result is None:
            continue
        cleaned.append(result)
        if result["extra"]["flagged_low_confidence"]:
            flagged.append(result)

    return cleaned, flagged


def main():
    parser = argparse.ArgumentParser(description="Clean raw OCR-extracted blocks")
    parser.add_argument("--in", dest="in_path", required=True, help="Path to raw OCR JSON")
    parser.add_argument("--out", dest="out_path", required=True, help="Path to save cleaned JSON")
    args = parser.parse_args()

    raw_blocks = load_blocks(args.in_path)
    cleaned, flagged = clean_blocks(raw_blocks)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Cleaned {len(cleaned)} blocks (from {len(raw_blocks)} raw) -> {args.out_path}")
    print(f"[INFO] {len(flagged)} blocks flagged as low-confidence for manual review.")


if __name__ == "__main__":
    main()
