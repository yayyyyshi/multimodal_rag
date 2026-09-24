"""Turns any supported file into cleaned ExtractedBlocks.

PDF: PyMuPDF for text, tables and figures, PaddleOCR for scanned pages only.
Image: one image block, with OCR text when OCR is enabled.
CSV/XLSX: one Markdown table block per sheet.
TXT/MD: one text block per paragraph.

Blocks are saved to <processed_dir>/<doc_id>/blocks.json. Images go to
<artifacts_dir>/images/<doc_id>/ and are referenced relative to artifacts_dir.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from src.extraction.pymupdf_extractor import extract_pymupdf_blocks
from src.preprocessing.cleaner import clean_blocks
from src.utils.config import resolve
from src.utils.metadata import IMAGE, TABLE, TEXT, ExtractedBlock, save_blocks

PDF_EXT = {".pdf"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TABLE_EXT = {".csv", ".xlsx", ".xls"}
TEXT_EXT = {".txt", ".md"}
SUPPORTED = PDF_EXT | IMG_EXT | TABLE_EXT | TEXT_EXT


def doc_id_from_path(path: str | Path) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", Path(path).stem).strip("_")


def rows_to_markdown(header: list, rows: list[list]) -> str:
    esc = lambda v: "" if v is None else str(v).replace("|", "/").replace("\n", " ").strip()
    lines = ["| " + " | ".join(esc(h) for h in header) + " |",
             "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(esc(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def _ingest_pdf(path: Path, doc_id: str, cfg: dict, image_dir: Path, artifacts_dir: Path) -> list[dict]:
    ecfg = cfg["extraction"]
    blocks, scanned = extract_pymupdf_blocks(path, doc_id, image_dir, artifacts_dir, ecfg)
    blocks = [b.to_dict() for b in blocks]
    if scanned:
        if ecfg.get("use_ocr", True):
            try:
                from src.extraction.ocr_extractor import extract_ocr_blocks
                ocr = extract_ocr_blocks(path, doc_id, pages=scanned, dpi=ecfg.get("ocr_dpi", 200),
                                         tmp_image_dir=resolve(cfg["paths"]["processed_dir"]) / "_tmp_pages")
                blocks += [b.to_dict() for b in ocr]
                page_text = {}
                for b in ocr:
                    page_text[b.page_num] = (page_text.get(b.page_num, "") + " " + b.text).strip()
                for b in blocks:  # copy the OCR text onto the full-page image block
                    if b["source_type"] == "pymupdf_page" and b["page_num"] in page_text:
                        b["extra"]["ocr_text"] = page_text[b["page_num"]][:1500]
                print(f"[ingest] {doc_id}: OCR on {len(scanned)} scanned page(s) -> {len(ocr)} blocks")
            except ImportError:
                print(f"[ingest] {doc_id}: {len(scanned)} scanned page(s) skipped (PaddleOCR not installed)")
        else:
            print(f"[ingest] {doc_id}: {len(scanned)} scanned page(s) skipped (use_ocr=false)")
    return blocks


def _ingest_image(path: Path, doc_id: str, cfg: dict, image_dir: Path, artifacts_dir: Path) -> list[dict]:
    image_dir.mkdir(parents=True, exist_ok=True)
    dst = image_dir / f"p001_img0{path.suffix.lower()}"
    shutil.copy(path, dst)
    text = ""
    if cfg["extraction"].get("use_ocr", True):
        try:
            from src.extraction.ocr_extractor import ocr_image_text
            text = ocr_image_text(str(dst))
        except ImportError:
            pass
    return [ExtractedBlock(doc_id=doc_id, page_num=1, text=text, source_type="image_file", modality=IMAGE,
                           extra={"image_path": dst.relative_to(artifacts_dir).as_posix(),
                                  "caption": path.stem.replace("_", " ")}).to_dict()]


def _ingest_table(path: Path, doc_id: str, **_) -> list[dict]:
    import pandas as pd
    sheets = pd.read_excel(path, sheet_name=None) if path.suffix.lower() != ".csv" else {"": pd.read_csv(path)}
    blocks = []
    for i, (name, df) in enumerate(sheets.items(), start=1):
        df = df.dropna(how="all").fillna("")
        md = rows_to_markdown(list(map(str, df.columns)), df.astype(str).values.tolist())
        blocks.append(ExtractedBlock(doc_id=doc_id, page_num=i, text=md, source_type="csv", modality=TABLE,
                                     extra={"caption": f"{path.stem} {name}".strip(), "n_rows": len(df)}).to_dict())
    return blocks


def _ingest_text(path: Path, doc_id: str, **_) -> list[dict]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    return [ExtractedBlock(doc_id=doc_id, page_num=1, text=p, source_type="text_file", modality=TEXT).to_dict()
            for p in paras]


def ingest_file(path: str | Path, cfg: dict, doc_id: str | None = None, save: bool = True) -> list[dict]:
    path = Path(path)
    doc_id = doc_id or doc_id_from_path(path)
    ext = path.suffix.lower()
    artifacts_dir = resolve(cfg["paths"]["artifacts_dir"])
    image_dir = artifacts_dir / "images" / doc_id
    if image_dir.exists():
        shutil.rmtree(image_dir)  # drop crops from a previous run

    kw = dict(doc_id=doc_id, cfg=cfg, image_dir=image_dir, artifacts_dir=artifacts_dir)
    if ext in PDF_EXT:
        blocks = _ingest_pdf(path, **kw)
    elif ext in IMG_EXT:
        blocks = _ingest_image(path, **kw)
    elif ext in TABLE_EXT:
        blocks = _ingest_table(path, doc_id=doc_id)
    elif ext in TEXT_EXT:
        blocks = _ingest_text(path, doc_id=doc_id)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    blocks, flagged = clean_blocks(blocks)
    for b in blocks:
        b.setdefault("extra", {})["source_file"] = path.name
    if save:
        out = resolve(cfg["paths"]["processed_dir"]) / doc_id / "blocks.json"
        save_blocks(blocks, out)
    counts = {m: sum(b.get("modality") == m for b in blocks) for m in (TEXT, TABLE, IMAGE)}
    print(f"[ingest] {doc_id}: {len(blocks)} blocks {counts}" + (f", {len(flagged)} low-confidence" if flagged else ""))
    return blocks


def discover_files(inputs: list[str | Path]) -> list[Path]:
    files = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED)
        elif p.suffix.lower() in SUPPORTED:
            files.append(p)
    return files
