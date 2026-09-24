"""Step 3, run locally: load the Kaggle artifacts into ChromaDB.

    python scripts/build_index.py --zip artifacts.zip   # unzip and index
    python scripts/build_index.py                       # artifacts/ already unzipped

No model is loaded, the vectors are read from the .npy files.
"""

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config, resolve
from src.vectorstore.chroma_store import build_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--zip", default=None, help="path to artifacts.zip (unzipped into the project root first)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.zip:
        with zipfile.ZipFile(args.zip) as z:
            z.extractall(resolve("."))
        print(f"[index] extracted {args.zip}")

    art = resolve(cfg["paths"]["artifacts_dir"])
    manifest = build_index(art, resolve(cfg["paths"]["chroma_dir"]))

    m = cfg["models"]
    for key, cfg_key in (("text_embedder", "text_embedder"), ("clip", "clip")):
        if manifest[key] != m[cfg_key]:
            print(f"[WARNING] artifacts were built with {key}={manifest[key]!r} but config says {m[cfg_key]!r}. "
                  f"Queries will use the artifact's model (from manifest.json).")
    print("[index] ready. Try:  python scripts/ask.py \"your question\"")


if __name__ == "__main__":
    main()
