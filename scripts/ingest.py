"""Step 1: extract text, tables and figures, then chunk them.

    python scripts/ingest.py --subset              # documents in data/eval/subset_docs.json
    python scripts/ingest.py --inputs data/raw     # own PDF, image, CSV/XLSX, TXT/MD files
    python scripts/ingest.py --subset --no-ocr     # skip PaddleOCR

Writes data/processed/<doc_id>/blocks.json, artifacts/images/<doc_id>/*.jpg
and artifacts/chunks.jsonl.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.chunking.chunker import chunk_document
from src.ingestion.pipeline import discover_files, doc_id_from_path, ingest_file
from src.utils.config import load_config, resolve
from src.utils.metadata import save_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--subset", action="store_true", help="ingest the MMLongBench-Doc subset")
    ap.add_argument("--inputs", nargs="*", default=[], help="files or folders")
    ap.add_argument("--no-ocr", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, {"extraction": {"use_ocr": False}} if args.no_ocr else None)

    jobs: list[tuple[Path, str]] = []
    if args.subset:
        for d in json.loads(resolve("data/eval/subset_docs.json").read_text(encoding="utf-8")):
            jobs.append((resolve(d["pdf_path"]), d["doc_id"]))
    for f in discover_files(args.inputs):
        jobs.append((f, doc_id_from_path(f)))
    if not jobs:
        ap.error("nothing to ingest: pass --subset and/or --inputs")

    all_chunks, t0 = [], time.time()
    for i, (path, doc_id) in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] {path.name}")
        try:
            blocks = ingest_file(path, cfg, doc_id=doc_id)
        except Exception as e:
            print(f"[ingest] FAILED {path}: {e}")
            continue
        all_chunks += chunk_document(blocks, cfg["chunking"])

    out = resolve(cfg["paths"]["artifacts_dir"]) / "chunks.jsonl"
    save_jsonl(all_chunks, out)
    by = {}
    for c in all_chunks:
        by[c.modality] = by.get(c.modality, 0) + 1
    print(f"\n[DONE] {len(jobs)} files -> {len(all_chunks)} chunks {by} in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
