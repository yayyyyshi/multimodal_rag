"""Loads MMLongBench-Doc and selects the document subset.

MMLongBench-Doc (Ma et al., NeurIPS 2024) has 135 PDFs and 1,082 questions.
Each question has evidence_pages, evidence_sources (Pure-text, Layout, Table,
Chart, Figure), answer_format (Str, Int, Float, List, None) and an answer,
which is "Not answerable" for about 20% of them.

Source: https://github.com/mayubo2333/MMLongBench-Doc. PDFs are in
data/documents/ and questions in data/samples.json.
"""

from __future__ import annotations

import ast
import json
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path

NOT_ANSWERABLE = "Not answerable"


def make_doc_id(filename: str) -> str:
    """'PH_2016.06.08_Economy-Final.pdf' -> 'PH_2016_06_08_Economy_Final'."""
    stem = Path(filename).stem
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")


def clone_mmlongbench(repo_url: str, local_dir: Path) -> Path:
    """Shallow clone (about 640 MB), skipped if samples.json already exists."""
    if (local_dir / "data" / "samples.json").exists():
        return local_dir
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[dataset] cloning {repo_url} -> {local_dir} (about 640 MB)")
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(local_dir)], check=True)
    return local_dir


def _parse_list(s):
    if isinstance(s, list):
        return s
    try:
        v = ast.literal_eval(s)
        return list(v) if isinstance(v, (list, tuple)) else [v]
    except Exception:
        return []


def load_samples(local_dir: Path) -> list[dict]:
    with open(local_dir / "data" / "samples.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for i, s in enumerate(raw):
        out.append({
            "qid": f"q{i:04d}",
            "doc_file": s["doc_id"],
            "doc_id": make_doc_id(s["doc_id"]),
            "doc_type": s["doc_type"],
            "question": s["question"],
            "answer": str(s["answer"]),
            "answer_format": s["answer_format"],
            "evidence_pages": [int(p) for p in _parse_list(s["evidence_pages"])],
            "evidence_sources": [str(x) for x in _parse_list(s["evidence_sources"])],
            "answerable": str(s["answer"]).strip() != NOT_ANSWERABLE,
        })
    return out


def page_count(pdf_path: Path) -> int:
    import pymupdf
    with pymupdf.open(pdf_path) as d:
        return len(d)


def select_subset(samples: list[dict], docs_dir: Path, n_docs: int, max_pages: int, seed: int = 42) -> list[str]:
    """Returns n_docs PDF filenames, taken round-robin across doc_type.

    Documents longer than max_pages are skipped. Within a type, documents whose
    questions cover more evidence types come first.
    """
    rng = random.Random(seed)
    by_doc = defaultdict(list)
    for s in samples:
        by_doc[s["doc_file"]].append(s)

    by_type = defaultdict(list)
    for doc_file, qs in by_doc.items():
        pdf = docs_dir / doc_file
        if not pdf.exists():
            continue
        try:
            if page_count(pdf) > max_pages:
                continue
        except Exception:
            continue
        kinds = {src for q in qs for src in q["evidence_sources"]}
        score = len(kinds) * 10 + len(qs) + rng.random()
        by_type[qs[0]["doc_type"]].append((score, doc_file))

    for t in by_type:
        by_type[t].sort(reverse=True)

    chosen, types = [], sorted(by_type)
    while len(chosen) < n_docs and any(by_type.values()):
        for t in types:
            if by_type[t] and len(chosen) < n_docs:
                chosen.append(by_type[t].pop(0)[1])
    return chosen
