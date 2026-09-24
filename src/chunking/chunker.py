"""Splits cleaned blocks into Chunks for retrieval.

Text blocks of a page are packed into chunks of up to max_chars with a small
overlap. Chunks never span two pages, so every citation points to one page.
Tables become one chunk each; long tables are split every
max_table_rows_per_chunk rows with the header row repeated.
Each figure becomes one chunk described by its caption and inner text. The
BLIP caption and OCR text are added later on Kaggle by build_embeddings.py.

The embedding text of every chunk starts with "Document: <title> | Page N |
<type>" so short generic chunks still carry their document context.
"""

from __future__ import annotations

import re
from collections import defaultdict

from src.utils.metadata import IMAGE, TABLE, TEXT, Chunk

DEFAULTS = {"max_chars": 1000, "overlap_chars": 150, "max_table_rows_per_chunk": 25, "add_context_header": True}


def guess_title(blocks: list[dict], doc_id: str) -> str:
    for b in blocks:
        if b.get("modality") == TEXT and b["page_num"] == min(x["page_num"] for x in blocks):
            t = b["text"].strip()
            if 8 <= len(t) and not t.lower().startswith(("www.", "http")):
                return t[:120]
    return doc_id.replace("_", " ")


def context_header(title: str, page: int, modality: str) -> str:
    return f"Document: {title} | Page {page} | {modality}"


def split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Splits at sentence ends; each piece starts with `overlap` chars of the previous one."""
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces, cur = [], ""
    for s in sentences:
        while len(s) > max_chars:  # sentence longer than max_chars, cut it hard
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(s[:max_chars])
            s = s[max_chars:]
        if len(cur) + len(s) + 1 > max_chars and cur:
            pieces.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        pieces.append(cur)
    if overlap <= 0:
        return pieces
    out = [pieces[0]]
    for prev, p in zip(pieces, pieces[1:]):
        tail = prev[-overlap:]
        tail = tail[tail.find(" ") + 1:] if " " in tail else tail
        out.append(f"{tail} {p}")
    return out


def compose_image_content(extra: dict, inside_text: str, page: int) -> str:
    """Text description of a figure built from its caption, BLIP caption and inner text."""
    parts = [f"[Figure on page {page}]"]
    if extra.get("caption"):
        parts.append(f"Caption: {extra['caption']}")
    if extra.get("generated_caption"):
        parts.append(f"Visual description: {extra['generated_caption']}")
    txt = " ".join(t for t in [inside_text, extra.get("ocr_text", "")] if t).strip()
    if txt:
        parts.append(f"Text/values in figure: {txt[:1200]}")
    return "\n".join(parts)


def _pack_text_blocks(texts: list[str], max_chars: int, overlap: int) -> list[str]:
    chunks, cur = [], ""
    for t in texts:
        if len(cur) + len(t) + 1 <= max_chars:
            cur = f"{cur}\n{t}".strip()
            continue
        if cur:
            chunks.append(cur)
        if len(t) > max_chars:
            parts = split_long_text(t, max_chars, overlap)
            chunks += parts[:-1]
            cur = parts[-1]
        else:
            tail = chunks[-1][-overlap:] if (chunks and overlap) else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else ""
            cur = f"{tail} {t}".strip() if tail else t
    if cur:
        chunks.append(cur)
    return chunks


def _split_table(markdown: str, max_rows: int) -> list[str]:
    lines = markdown.strip().splitlines()
    if len(lines) <= max_rows + 2:
        return [markdown.strip()]
    header, body = lines[:2], lines[2:]
    return ["\n".join(header + body[i:i + max_rows]) for i in range(0, len(body), max_rows)]


def chunk_document(blocks: list[dict], cfg: dict | None = None) -> list[Chunk]:
    cfg = {**DEFAULTS, **(cfg or {})}
    if not blocks:
        return []
    doc_id = blocks[0]["doc_id"]
    title = guess_title(blocks, doc_id)
    use_header = cfg["add_context_header"]
    chunks: list[Chunk] = []
    counter = defaultdict(int)

    def add(page, modality, content, **kw):
        n = counter[(page, modality)]
        counter[(page, modality)] += 1
        head = context_header(title, page, modality) + "\n" if use_header else ""
        chunks.append(Chunk(
            chunk_id=f"{doc_id}::p{page}::{modality}::{n}", doc_id=doc_id, page_num=page, modality=modality,
            text=head + content, content=content, **kw,
        ))

    by_page = defaultdict(list)
    for b in blocks:
        by_page[b["page_num"]].append(b)

    for page in sorted(by_page):
        pb = by_page[page]
        texts = [b["text"] for b in pb if b.get("modality", TEXT) == TEXT and b["text"].strip()]
        for piece in _pack_text_blocks(texts, cfg["max_chars"], cfg["overlap_chars"]):
            add(page, TEXT, piece, source_type="text")

        for b in pb:
            extra = dict(b.get("extra") or {})
            if b.get("modality") == TABLE:
                cap = extra.get("caption", "")
                for piece in _split_table(b["text"], cfg["max_table_rows_per_chunk"]):
                    content = (f"Table caption: {cap}\n" if cap else "") + piece
                    add(page, TABLE, content, bbox=b.get("bbox"), source_type=b["source_type"], extra={"caption": cap})
            elif b.get("modality") == IMAGE:
                extra["inside_text"] = b.get("text", "")
                add(page, IMAGE, compose_image_content(extra, b.get("text", ""), page),
                    image_path=extra.get("image_path"), bbox=b.get("bbox"), source_type=b["source_type"], extra=extra)
    return chunks


def refresh_image_chunk(chunk: dict, title: str | None = None) -> dict:
    """Rebuilds text and content after OCR text or a BLIP caption was added to extra."""
    extra = chunk.get("extra", {})
    content = compose_image_content(extra, extra.get("inside_text", ""), chunk["page_num"])
    header = chunk["text"].split("\n", 1)[0] if chunk["text"].startswith("Document:") else None
    chunk["content"] = content
    chunk["text"] = f"{header}\n{content}" if header else content
    return chunk
