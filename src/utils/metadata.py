"""
Shared output schema for extracted document content.

IMPORTANT: This file must be IDENTICAL for both team members.
Jyoti's PyMuPDF extractor and Yashika's PaddleOCR extractor should
both produce output in this exact structure, so downstream chunking
and embedding code can treat both sources the same way.

Agree on this schema with your teammate BEFORE writing extraction
code, so your PRs don't conflict later.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class ExtractedBlock:
    """
    One unit of extracted content from a document (a text block,
    OCR line-group, or paragraph).
    """
    doc_id: str                     # e.g. filename without extension
    page_num: int                   # 1-indexed page number
    text: str                       # cleaned text content
    source_type: str                # "pymupdf_text" or "paddleocr"
    confidence: Optional[float] = None   # OCR confidence (0-1), None for direct text
    bbox: Optional[list] = None          # [x0, y0, x1, y1] bounding box, if available
    extra: dict = field(default_factory=dict)  # anything extra (e.g. rotation, language)

    def to_dict(self):
        return asdict(self)


def save_blocks(blocks: list[ExtractedBlock], out_path: str):
    """Save a list of ExtractedBlock objects to a JSON file."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([b.to_dict() for b in blocks], f, ensure_ascii=False, indent=2)


def load_blocks(in_path: str) -> list[dict]:
    """Load extracted blocks back from a JSON file."""
    with open(in_path, "r", encoding="utf-8") as f:
        return json.load(f)
