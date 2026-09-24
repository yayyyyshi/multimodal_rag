"""ChromaDB store built from the embeddings computed on Kaggle.

mm_text holds every chunk (text, table, figure description) with its text
embedding. mm_image holds every figure with its CLIP embedding. Both use
cosine distance. Building the index only reads the .npy files, no model is
loaded; models are needed only to embed the question at query time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.utils.metadata import load_jsonl

TEXT_COLLECTION = "mm_text"
IMAGE_COLLECTION = "mm_image"
_BATCH = 2000


def _meta(c: dict) -> dict:
    extra = c.get("extra") or {}
    return {
        "doc_id": c["doc_id"],
        "page_num": int(c["page_num"]),
        "modality": c["modality"],
        "image_path": c.get("image_path") or "",
        "source_type": c.get("source_type") or "",
        "caption": (extra.get("caption") or "")[:500],
    }


_CLIENTS: dict[str, object] = {}


def _client(chroma_dir: str | Path):
    """Reuses one client per folder; two clients on the same folder break Chroma reads."""
    import chromadb
    from chromadb.config import Settings
    key = str(Path(chroma_dir).resolve())
    if key not in _CLIENTS:
        Path(key).mkdir(parents=True, exist_ok=True)
        _CLIENTS[key] = chromadb.PersistentClient(path=key, settings=Settings(anonymized_telemetry=False))
    return _CLIENTS[key]


def build_index(artifacts_dir: str | Path, chroma_dir: str | Path) -> dict:
    """Deletes and recreates both collections. Returns the manifest."""
    art = Path(artifacts_dir)
    manifest = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
    chunks = load_jsonl(art / "chunks.jsonl")
    text_emb = np.load(art / "text_embeddings.npy")
    image_emb = np.load(art / "image_embeddings.npy")
    image_ids = json.loads((art / "image_chunk_ids.json").read_text(encoding="utf-8"))
    assert len(chunks) == len(text_emb), "chunks.jsonl and text_embeddings.npy have different lengths, rebuild the artifacts on Kaggle"
    assert len(image_ids) == len(image_emb), "image_chunk_ids.json and image_embeddings.npy are out of sync"

    client = _client(chroma_dir)
    for name in (TEXT_COLLECTION, IMAGE_COLLECTION):
        try:
            client.delete_collection(name)
        except Exception:
            pass
    coll_meta = {"hnsw:space": "cosine", "text_embedder": manifest["text_embedder"], "clip": manifest["clip"]}
    tcol = client.create_collection(TEXT_COLLECTION, metadata=coll_meta)
    icol = client.create_collection(IMAGE_COLLECTION, metadata=coll_meta)

    for s in range(0, len(chunks), _BATCH):
        part = chunks[s:s + _BATCH]
        tcol.add(ids=[c["chunk_id"] for c in part], embeddings=text_emb[s:s + _BATCH].tolist(),
                 documents=[c["content"] for c in part], metadatas=[_meta(c) for c in part])

    by_id = {c["chunk_id"]: c for c in chunks}
    for s in range(0, len(image_ids), _BATCH):
        ids = image_ids[s:s + _BATCH]
        icol.add(ids=ids, embeddings=image_emb[s:s + _BATCH].tolist(),
                 documents=[by_id[i]["content"] for i in ids], metadatas=[_meta(by_id[i]) for i in ids])
    print(f"[chroma] {tcol.count()} text-side vectors, {icol.count()} image vectors -> {chroma_dir}")
    return manifest


class ChromaStore:
    def __init__(self, chroma_dir: str | Path, exact_search: bool = False):
        self.use_exact = exact_search
        self._cache: dict = {}
        client = _client(chroma_dir)
        try:
            self.text = client.get_collection(TEXT_COLLECTION)
            self.image = client.get_collection(IMAGE_COLLECTION)
        except Exception as e:
            raise RuntimeError(f"No index in {chroma_dir}. Run: python scripts/build_index.py") from e
        self.meta = self.text.metadata or {}

    @staticmethod
    def _where(doc_ids: list[str] | None, modality: str | None = None):
        conds = []
        if doc_ids:
            conds.append({"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": list(doc_ids)}})
        if modality:
            conds.append({"modality": modality})
        if not conds:
            return None
        return conds[0] if len(conds) == 1 else {"$and": conds}

    @staticmethod
    def _unpack(res) -> list[dict]:
        out = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append({"chunk_id": cid, "content": doc, **meta, "score": 1.0 - float(dist)})
        return out

    def _exact_search(self, col, vec: np.ndarray, k: int, where) -> list[dict]:
        """Brute-force cosine search in numpy over the stored vectors.

        Fallback for the chromadb 1.x error on Windows "Error creating hnsw
        segment reader: Nothing found on disk". With a few thousand vectors
        this still takes milliseconds.
        """
        key = (col.name, json.dumps(where, sort_keys=True))
        if key not in self._cache:
            got = col.get(where=where, include=["embeddings", "documents", "metadatas"])
            emb = np.asarray(got["embeddings"], dtype=np.float32) if len(got["ids"]) else np.zeros((0, 1), np.float32)
            self._cache[key] = (got["ids"], got["documents"], got["metadatas"], emb)
        ids, docs, metas, emb = self._cache[key]
        if not len(ids):
            return []
        sims = emb @ np.asarray(vec, dtype=np.float32)
        top = np.argsort(-sims)[:k]
        return [{"chunk_id": ids[i], "content": docs[i], **metas[i], "score": float(sims[i])} for i in top]

    def _search(self, col, vec, k: int, where) -> list[dict]:
        n = col.count()
        if n == 0:
            return []
        if not self.use_exact:
            try:
                res = col.query(query_embeddings=[np.asarray(vec).tolist()], n_results=min(k, n), where=where)
                return self._unpack(res)
            except Exception as e:
                print(f"[chroma] HNSW query failed ({str(e)[:80]}...) -> using exact numpy search")
                self.use_exact = True
        return self._exact_search(col, vec, k, where)

    def query_text(self, vec: np.ndarray, k: int, doc_ids=None, modality=None) -> list[dict]:
        return self._search(self.text, vec, k, self._where(doc_ids, modality))

    def query_image(self, vec: np.ndarray, k: int, doc_ids=None) -> list[dict]:
        return self._search(self.image, vec, k, self._where(doc_ids))

    def list_docs(self) -> list[str]:
        metas = self.text.get(include=["metadatas"])["metadatas"]
        return sorted({m["doc_id"] for m in metas})
