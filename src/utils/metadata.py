"""Data classes shared by all pipeline stages, plus JSON/JSONL helpers.

ExtractedBlock is what the extractors produce. Chunk is what gets embedded,
stored in ChromaDB and cited in answers. `modality` has a default so older
block files in data/processed/ still load.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

TEXT, TABLE, IMAGE = "text", "table", "image"


@dataclass
class ExtractedBlock:
    doc_id: str
    page_num: int                        # 1-indexed
    text: str                            # for images: text printed inside the figure
    source_type: str                     # pymupdf_text, paddleocr, pymupdf_table, pymupdf_image, pymupdf_figure, ...
    confidence: Optional[float] = None   # OCR score 0-1, None for native text
    bbox: Optional[list] = None          # [x0, y0, x1, y1] in PDF points
    extra: dict = field(default_factory=dict)   # caption, image_path, flags
    modality: str = TEXT                 # text, table or image

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractedBlock":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class Chunk:
    chunk_id: str                        # <doc_id>::p<page>::<modality>::<n>
    doc_id: str
    page_num: int
    modality: str
    text: str                            # input to the text embedder, starts with the context header
    content: str                         # passed to the LLM and shown in the UI
    image_path: Optional[str] = None     # relative to artifacts_dir
    bbox: Optional[list] = None
    source_type: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def save_blocks(blocks: list, out_path: str | Path):
    """Accepts ExtractedBlock objects or plain dicts."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    data = [b.to_dict() if hasattr(b, "to_dict") else b for b in blocks]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_blocks(in_path: str | Path) -> list[dict]:
    with open(in_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(rows: list, out_path: str | Path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict() if hasattr(r, "to_dict") else r, ensure_ascii=False) + "\n")


def load_jsonl(in_path: str | Path) -> list[dict]:
    with open(in_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
