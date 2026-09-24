"""Step 2, run on the Kaggle GPU: caption and OCR the figures, then embed everything.

    python scripts/build_embeddings.py
    python scripts/build_embeddings.py --ocr-figures        # OCR charts without a text layer
    python scripts/build_embeddings.py --no-caption --zip   # skip captions, write artifacts.zip

Reads artifacts/chunks.jsonl and artifacts/images/ (from ingest.py) and writes:
    chunks.jsonl            figure chunks updated with the generated caption and OCR text
    text_embeddings.npy     row i belongs to line i of chunks.jsonl
    image_embeddings.npy    one CLIP vector per figure
    image_chunk_ids.json    chunk_id of each row in image_embeddings.npy
    manifest.json           model names and dimensions, read again at query time
"""

import argparse
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.chunking.chunker import refresh_image_chunk
from src.embeddings.models import ClipEmbedder, TextEmbedder
from src.utils.config import load_config, resolve
from src.utils.metadata import load_jsonl, save_jsonl


def zip_artifacts(art: Path) -> Path:
    out = art.parent / "artifacts.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in art.rglob("*"):
            if f.is_file() and "chroma" not in f.parts and f.name != ".gitkeep":
                z.write(f, f.relative_to(art.parent))
    print(f"[zip] {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-caption", action="store_true", help="skip figure captions")
    ap.add_argument("--ocr-figures", action="store_true", help="PaddleOCR on figures that have no text layer")
    ap.add_argument("--zip", action="store_true")
    ap.add_argument("--text-model", default=None, help="override models.text_embedder (e.g. hash:384 for a smoke test)")
    ap.add_argument("--clip-model", default=None, help="override models.clip")
    args = ap.parse_args()

    cfg = load_config(args.config)
    m = cfg["models"]
    text_model = args.text_model or m["text_embedder"]
    clip_model = args.clip_model or m["clip"]
    art = resolve(cfg["paths"]["artifacts_dir"])
    chunks = load_jsonl(art / "chunks.jsonl")
    img_idx = [i for i, c in enumerate(chunks) if c["modality"] == "image" and c.get("image_path")
               and (art / c["image_path"]).exists()]
    print(f"[embed] {len(chunks)} chunks, {len(img_idx)} images")
    t0 = time.time()

    # chart labels drawn as paths have no text layer, so OCR the crop
    if args.ocr_figures:
        from src.extraction.ocr_extractor import ocr_image_text
        todo = [i for i in img_idx if len(chunks[i]["extra"].get("inside_text", "")) < 20
                and not chunks[i]["extra"].get("ocr_text")]
        print(f"[embed] OCR on {len(todo)} figures ...")
        for n, i in enumerate(todo, 1):
            chunks[i]["extra"]["ocr_text"] = ocr_image_text(str(art / chunks[i]["image_path"]))[:1500]
            if n % 50 == 0:
                print(f"  {n}/{len(todo)}")

    backend = m.get("captioner_backend", "blip")
    if not args.no_caption and img_idx:
        if backend == "qwen_vl":
            from src.embeddings.models import VLCaptioner
            todo = img_idx
            cap = VLCaptioner(m["captioner_vl"])
        else:
            # BLIP only describes pictures well; for text-heavy figures (tables,
            # code, diagrams full of labels) the inner text is already better
            from src.embeddings.models import Captioner
            todo = [i for i in img_idx if len(chunks[i]["extra"].get("inside_text", "")) < 150]
            cap = Captioner(m["captioner"])
        print(f"[embed] {backend} captions for {len(todo)} figures ...")
        caps = cap.caption([str(art / chunks[i]["image_path"]) for i in todo])
        for i, c in zip(todo, caps):
            chunks[i]["extra"]["generated_caption"] = c
        if hasattr(cap, "unload"):
            cap.unload()
        del cap
        print(f"[embed] captions done ({time.time() - t0:.0f}s)")

    for i in img_idx:
        refresh_image_chunk(chunks[i])

    # text embeddings for every chunk, figures included through their description
    te = TextEmbedder(text_model, m.get("text_query_instruction", ""))
    text_emb = te.embed_documents([c["text"] for c in chunks], show_progress=True)
    print(f"[embed] text embeddings {text_emb.shape} ({time.time() - t0:.0f}s)")

    # CLIP image embeddings
    clip = ClipEmbedder(clip_model)
    image_emb = clip.embed_images([str(art / chunks[i]["image_path"]) for i in img_idx])
    print(f"[embed] image embeddings {image_emb.shape} ({time.time() - t0:.0f}s)")

    save_jsonl(chunks, art / "chunks.jsonl")
    np.save(art / "text_embeddings.npy", text_emb.astype(np.float32))
    np.save(art / "image_embeddings.npy", image_emb.astype(np.float32))
    (art / "image_chunk_ids.json").write_text(json.dumps([chunks[i]["chunk_id"] for i in img_idx]), encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "text_embedder": text_model, "text_dim": int(text_emb.shape[1]),
        "text_query_instruction": m.get("text_query_instruction", ""),
        "clip": clip_model, "clip_dim": int(image_emb.shape[1]) if len(img_idx) else clip.dim,
        "captioner": None if args.no_caption else (m["captioner_vl"] if backend == "qwen_vl" else m["captioner"]),
        "figure_ocr": bool(args.ocr_figures),
        "n_chunks": len(chunks), "n_images": len(img_idx),
        "n_docs": len({c["doc_id"] for c in chunks}),
    }
    (art / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if args.zip:
        zip_artifacts(art)


if __name__ == "__main__":
    main()
