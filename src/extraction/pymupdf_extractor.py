"""Extracts text, tables and figures from PDF pages that have a text layer.

Text blocks inside a detected table or figure are skipped so they are not
stored twice. Tables are saved as Markdown. Figures are either embedded raster
images or clusters of vector drawings (charts); each one is cropped to a JPEG
and keeps its nearby caption and any text printed inside it.

Pages with almost no text are returned as scanned pages so that only those
pages are sent to OCR.

Usage:
    python src/extraction/pymupdf_extractor.py --pdf data/raw/check.pdf --doc-id check
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

import pymupdf as fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.metadata import IMAGE, TABLE, TEXT, ExtractedBlock, save_blocks

CAPTION_RE = re.compile(r"^\s*(fig(ure)?|table|chart|exhibit|graph)\s*[\dIVX]*[.:]?", re.IGNORECASE)

DEFAULTS = {
    "min_text_chars_native": 30,
    "figure_dpi": 150,
    "min_image_side_pt": 60,
    "min_figure_area_ratio": 0.03,
    "max_figure_area_ratio": 0.95,
}


def _rect_overlap_ratio(inner: fitz.Rect, outer: fitz.Rect) -> float:
    """Share of inner's area that lies inside outer."""
    inter = fitz.Rect(inner) & outer
    if inter.is_empty or inner.get_area() == 0:
        return 0.0
    return inter.get_area() / inner.get_area()


def _merge_rects(rects: list[fitz.Rect], pad: float = 4.0) -> list[fitz.Rect]:
    """Merge touching rectangles so a chart and its legend become one region."""
    rects = [fitz.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        out: list[fitz.Rect] = []
        while rects:
            r = rects.pop()
            grown = fitz.Rect(r.x0 - pad, r.y0 - pad, r.x1 + pad, r.y1 + pad)
            i = 0
            while i < len(rects):
                if grown.intersects(rects[i]):
                    r |= rects.pop(i)
                    grown = fitz.Rect(r.x0 - pad, r.y0 - pad, r.x1 + pad, r.y1 + pad)
                    changed = True
                else:
                    i += 1
            out.append(r)
        rects = out
    return rects


def _clean(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)  # join words hyphenated across lines
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _caption_with_continuation(start: tuple, text_blocks: list[tuple], max_lines: int = 4) -> str:
    """A caption wrapped over several lines can be split into several blocks;
    append the blocks that follow directly below it (gap under 4 pt)."""
    parts, cur = [start[4]], start
    for _ in range(max_lines):
        nxt = [b for b in text_blocks if -2 <= b[1] - cur[3] < 4 and b is not cur and min(b[2], cur[2]) - max(b[0], cur[0]) > 0
               and not CAPTION_RE.match(b[4].strip())]
        if not nxt:
            break
        cur = min(nxt, key=lambda b: b[1])
        parts.append(cur[4])
    return _clean(" ".join(parts))


def _find_caption(rect: fitz.Rect, text_blocks: list[tuple], max_gap: float = 40.0, prefer: str = "fig") -> str:
    """Prefer a block starting with 'Figure N' or 'Table N' that is inside the
    region or within max_gap of it; captions of the preferred kind ('fig' or
    'table') win over closer ones of the other kind. If none exists, use up to
    two short blocks directly above, since reports put chart titles on top."""
    best, best_key = None, (2, max_gap)
    above: list[tuple[float, str]] = []
    for b in text_blocks:
        x0, y0, x1, y1, txt = b[:5]
        t = txt.strip()
        horiz_overlap = min(x1, rect.x1) - max(x0, rect.x0)
        if horiz_overlap <= 0:
            continue
        inside = _rect_overlap_ratio(fitz.Rect(x0, y0, x1, y1), rect) > 0.8
        gap = 0.0 if inside else (y0 - rect.y1 if y0 >= rect.y1 else rect.y0 - y1)
        if CAPTION_RE.match(t) and 0 <= gap < max_gap:
            key = (0 if t.lower().startswith(prefer) else 1, gap)
            if key < best_key:
                best, best_key = b, key
        elif not inside and y1 <= rect.y0 + 2 and 0 <= gap < 70 and len(t) < 300:
            above.append((gap, _clean(t)))
    if best is not None:
        return _caption_with_continuation(best, text_blocks)
    above.sort()
    return " ".join(t for _, t in reversed(above[:2]))


def _looks_like_table(caption: str, inside_text: str) -> bool:
    """Tables drawn with horizontal rules only are missed by find_tables() and
    end up as drawing clusters. Only the caption is trusted here: bar charts
    are also full of numbers but must stay images for CLIP."""
    return bool(re.match(r"^\s*table\s*[\dIVX]+", caption, re.IGNORECASE)) and len(inside_text) > 40


def _numbers_intact(rows: list, page_words: set[str]) -> bool:
    """strategy='text' can split '53.40' into '5340' and '.'; every number must exist on the page."""
    nums = [w for row in rows for c in row if c for w in str(c).split() if re.search(r"\d", w)]
    return all(w.strip("().,;:%$*") in page_words or w in page_words for w in nums)


def _table_text_in_region(page: fitz.Page, rect: fitz.Rect, page_words: set[str]) -> str:
    """Markdown from find_tables(strategy='text') inside rect, or '' if it is not usable."""
    try:
        tabs = page.find_tables(clip=rect, strategy="text")
    except Exception:
        return ""
    for t in tabs.tables:
        try:
            md, rows = t.to_markdown(clean=True).strip(), t.extract()
        except Exception:
            continue
        md = re.sub(r"<br\s*/?>", " ", html.unescape(html.unescape(md)))
        if t.row_count >= 2 and t.col_count >= 2 and _table_is_plausible({"rows": rows}, page_words) \
                and _numbers_intact(rows, page_words):
            return md
    return ""


def _table_is_plausible(table: dict, page_words: set[str]) -> bool:
    """find_tables() also detects bar charts and boxed text as tables. In those
    false hits the cells hold word fragments ('abou') or doubled glyphs
    ('GGeenneerraall'), so we require 80% of cell words to exist on the page."""
    cells = [str(c) for row in table["rows"] for c in row if c not in (None, "")]
    n_total = sum(len(row) for row in table["rows"]) or 1
    if len(cells) / n_total < 0.5:
        return False
    tokens = [w for c in cells for w in re.split(r"\s+", c) if len(w) > 2]
    if not tokens:
        return True  # numbers only
    ok = sum(w.strip(".,;:()%$") in page_words or w in page_words for w in tokens)
    return ok / len(tokens) >= 0.8


def extract_tables(page: fitz.Page) -> list[dict]:
    out = []
    try:
        tabs = page.find_tables()
    except Exception:
        return out
    page_words = {w[4].strip(".,;:()%$") for w in page.get_text("words")} | {w[4] for w in page.get_text("words")}
    parts = []
    for t in tabs.tables:
        try:
            md = t.to_markdown(clean=True).strip()
            rows = t.extract()
        except Exception:
            continue
        if not md or t.col_count < 2:
            continue
        md = re.sub(r"<br\s*/?>", " ", html.unescape(html.unescape(md)))
        parts.append({"bbox": fitz.Rect(t.bbox), "markdown": md, "rows": rows})

    # a table with horizontal rules between row groups comes back as several
    # tables stacked on top of each other with the same width; join them
    merged: list[dict] = []
    for p in sorted(parts, key=lambda p: p["bbox"].y0):
        last = merged[-1] if merged else None
        if last and abs(p["bbox"].x0 - last["bbox"].x0) < 3 and abs(p["bbox"].x1 - last["bbox"].x1) < 3 \
                and 0 <= p["bbox"].y0 - last["bbox"].y1 < 8:
            last["markdown"] += "\n" + "\n".join(p["markdown"].splitlines()[2:])  # drop repeated header lines
            last["rows"] += p["rows"]
            last["bbox"] |= p["bbox"]
        else:
            merged.append(p)

    for table in merged:
        if len(table["rows"]) < 2:
            continue
        if any(CAPTION_RE.match(str(c)) and str(c).lower().startswith("fig") for row in table["rows"] for c in row if c):
            continue  # a figure whose caption fell inside the detected grid
        if _table_is_plausible(table, page_words):
            out.append(table)
    return out


def find_visual_regions(page: fitz.Page, cfg: dict, table_rects: list[fitz.Rect]) -> list[tuple[fitz.Rect, str]]:
    """Returns (rect, kind) pairs, kind is 'pymupdf_image' or 'pymupdf_figure'."""
    page_area = page.rect.get_area()
    min_side = cfg["min_image_side_pt"]
    regions: list[tuple[fitz.Rect, str]] = []

    for info in page.get_image_info():
        r = fitz.Rect(info["bbox"]) & page.rect
        if r.is_empty or r.width < min_side or r.height < min_side:
            continue
        if r.get_area() / page_area > cfg["max_figure_area_ratio"]:
            continue  # page background or scanned page
        regions.append((r, "pymupdf_image"))

    # vector drawings: charts and diagrams
    try:
        clusters = page.cluster_drawings()
    except Exception:
        clusters = []
    for r in _merge_rects(clusters):
        ratio = r.get_area() / page_area
        if ratio < cfg["min_figure_area_ratio"] or ratio > cfg["max_figure_area_ratio"]:
            continue
        if any(_rect_overlap_ratio(r, t) > 0.5 or _rect_overlap_ratio(t, r) > 0.6 for t in table_rects):
            continue  # rules or frame of a table already found
        if any(_rect_overlap_ratio(r, img) > 0.7 for img, _ in regions):
            continue  # frame around an embedded image
        regions.append((r, "pymupdf_figure"))
    return regions


MAX_SIDE_PX = 1280


def save_crop(page: fitz.Page, rect: fitz.Rect, out_path: Path, dpi: int) -> None:
    """JPEG with the longest side capped, to keep the Kaggle download small."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    longest_pt = max(rect.width, rect.height, 1)
    dpi = min(dpi, int(MAX_SIDE_PX * 72 / longest_pt))
    pix = page.get_pixmap(clip=rect, dpi=max(dpi, 36), alpha=False)
    pix.save(str(out_path), jpg_quality=85)


def extract_pymupdf_blocks(
    pdf_path: str | Path,
    doc_id: str,
    image_dir: str | Path | None = None,
    image_rel_root: str | Path | None = None,
    cfg: dict | None = None,
) -> tuple[list[ExtractedBlock], list[int]]:
    """Returns (blocks, scanned_page_numbers).

    image_dir: where crops are written; None skips figure extraction.
    image_rel_root: image paths are stored relative to this folder (the
    artifacts folder) so the bundle works after moving it from Kaggle.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    doc = fitz.open(pdf_path)
    blocks: list[ExtractedBlock] = []
    scanned: list[int] = []
    image_dir = Path(image_dir) if image_dir else None

    for page_num, page in enumerate(doc, start=1):
        raw = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]  # b[6] == 1 is an image block
        total_chars = sum(len(b[4].strip()) for b in raw)
        if total_chars < cfg["min_text_chars_native"]:
            scanned.append(page_num)
            # keep the whole page as one image so CLIP can still match it; OCR adds the text
            if image_dir is not None:
                out = image_dir / f"p{page_num:03d}_page.jpg"
                save_crop(page, page.rect, out, cfg["figure_dpi"])
                try:
                    rel = out.resolve().relative_to(Path(image_rel_root).resolve()) if image_rel_root else out
                except ValueError:
                    rel = out
                blocks.append(ExtractedBlock(
                    doc_id=doc_id, page_num=page_num, text="", source_type="pymupdf_page",
                    bbox=[round(v, 2) for v in page.rect], modality=IMAGE,
                    extra={"image_path": Path(rel).as_posix(), "caption": f"Full page {page_num} (scanned/image-only)"},
                ))
            continue

        tables = extract_tables(page) if total_chars else []
        table_rects = [t["bbox"] for t in tables]
        for ti, t in enumerate(tables):
            blocks.append(ExtractedBlock(
                doc_id=doc_id, page_num=page_num, text=t["markdown"], source_type="pymupdf_table",
                bbox=[round(v, 2) for v in t["bbox"]], modality=TABLE,
                extra={"n_rows": len(t["rows"]), "caption": _find_caption(t["bbox"], raw, max_gap=80, prefer="table")},
            ))

        visual_rects: list[fitz.Rect] = []
        if image_dir is not None:
            for vi, (rect, kind) in enumerate(find_visual_regions(page, cfg, table_rects)):
                fname = f"p{page_num:03d}_{'img' if kind == 'pymupdf_image' else 'fig'}{vi}.jpg"
                out = image_dir / fname
                try:
                    save_crop(page, rect, out, cfg["figure_dpi"])
                except Exception as e:  # some PDFs have broken image streams
                    print(f"[PyMuPDF] could not crop {fname}: {e}")
                    continue
                visual_rects.append(rect)
                inside = _clean(page.get_text("text", clip=rect))
                caption = _find_caption(rect, raw)
                try:
                    rel = out.resolve().relative_to(Path(image_rel_root).resolve()) if image_rel_root else out
                except ValueError:
                    rel = out
                extra = {"image_path": Path(rel).as_posix(), "caption": caption}
                if kind == "pymupdf_figure" and _looks_like_table(caption, inside):
                    words = {w[4] for w in page.get_text("words")}
                    md = _table_text_in_region(page, rect, words | {w.strip(".,;:()%$") for w in words})
                    if any(b.page_num == page_num and b.modality == TABLE and b.text == (md or inside[:3000]) for b in blocks):
                        continue  # same table found through a second drawing cluster
                    blocks.append(ExtractedBlock(
                        doc_id=doc_id, page_num=page_num, text=md or inside[:3000], source_type="pymupdf_ruled_table",
                        bbox=[round(v, 2) for v in rect], modality=TABLE, extra=extra,
                    ))
                    continue
                blocks.append(ExtractedBlock(
                    doc_id=doc_id, page_num=page_num, text=inside[:1500], source_type=kind,
                    bbox=[round(v, 2) for v in rect], modality=IMAGE, extra=extra,
                ))

        for (x0, y0, x1, y1, txt, *_r) in raw:
            r = fitz.Rect(x0, y0, x1, y1)
            if any(_rect_overlap_ratio(r, t) > 0.6 for t in table_rects):
                continue
            if any(_rect_overlap_ratio(r, v) > 0.8 for v in visual_rects) and not CAPTION_RE.match(txt):
                continue
            clean = _clean(txt)
            if clean:
                blocks.append(ExtractedBlock(
                    doc_id=doc_id, page_num=page_num, text=clean, source_type="pymupdf_text",
                    bbox=[round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)], modality=TEXT,
                ))

    doc.close()
    return blocks, scanned


def main():
    ap = argparse.ArgumentParser(description="PyMuPDF extraction (text + tables + figures)")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--out", default=None, help="default: data/processed/<doc_id>/pymupdf.json")
    ap.add_argument("--image-dir", default=None, help="default: artifacts/images/<doc_id>")
    args = ap.parse_args()

    out = Path(args.out or f"data/processed/{args.doc_id}/pymupdf.json")
    img_dir = Path(args.image_dir or f"artifacts/images/{args.doc_id}")
    blocks, scanned = extract_pymupdf_blocks(args.pdf, args.doc_id, img_dir, image_rel_root="artifacts")
    save_blocks(blocks, out)
    by = {m: sum(b.modality == m for b in blocks) for m in (TEXT, TABLE, IMAGE)}
    print(f"[DONE] {len(blocks)} blocks {by} -> {out}")
    if scanned:
        print(f"[INFO] likely scanned pages (send to OCR): {scanned}")


if __name__ == "__main__":
    main()