"""Step 0: download MMLongBench-Doc and select the document subset.

    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --n-docs 10

Writes data/eval/subset_docs.json (doc_id, doc_file, pdf_path) and
data/eval/questions.jsonl (questions for those documents).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.dataset import clone_mmlongbench, load_samples, make_doc_id, select_subset
from src.utils.config import load_config, resolve
from src.utils.metadata import save_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-docs", type=int, default=None)
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--local-dir", default=None, help="where to clone the benchmark (e.g. /kaggle/temp/mmlongbench)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["dataset"]
    n_docs = args.n_docs or dcfg["n_docs"]
    max_pages = args.max_pages or dcfg["max_pages"]

    local_dir = clone_mmlongbench(dcfg["repo_url"], resolve(args.local_dir or dcfg["local_dir"]))
    docs_dir = local_dir / "data" / "documents"
    samples = load_samples(local_dir)

    chosen = select_subset(samples, docs_dir, n_docs, max_pages, dcfg["seed"])
    chosen_set = set(chosen)
    questions = [s for s in samples if s["doc_file"] in chosen_set]

    out_dir = resolve("data/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    def rel(p: Path) -> str:  # project-relative if possible, so the file works on other machines
        for a, b in ((p, resolve(".")), (p.resolve(), resolve(".").resolve())):
            try:
                return a.relative_to(b).as_posix()
            except ValueError:
                pass
        return str(p.resolve())

    subset = [{"doc_id": make_doc_id(f), "doc_file": f, "pdf_path": rel(docs_dir / f)} for f in chosen]
    (out_dir / "subset_docs.json").write_text(json.dumps(subset, indent=2), encoding="utf-8")
    save_jsonl(questions, out_dir / "questions.jsonl")

    print(f"[dataset] {len(chosen)} documents, {len(questions)} questions "
          f"({sum(not q['answerable'] for q in questions)} not answerable)")
    ev = Counter(src for q in questions for src in q["evidence_sources"])
    print("[dataset] evidence types:", dict(ev))
    print(f"[dataset] -> {out_dir / 'subset_docs.json'}\n[dataset] -> {out_dir / 'questions.jsonl'}")


if __name__ == "__main__":
    main()
