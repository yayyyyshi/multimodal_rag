"""Wrappers for the embedding, captioning and reranking models.

TextEmbedder: Sentence-Transformers, default bge-small-en-v1.5 (384-d).
ClipEmbedder: CLIP ViT-B/32 (512-d). Images and text share one space, so a
text query can retrieve images.
Captioner: BLIP image captioning.
Reranker: cross-encoder, default bge-reranker-base.

A model name of "hash:<dim>" gives HashingEmbedder instead. It needs no
download and is only meant for tests.

Torch and transformers are imported inside the classes, so modules that only
need the store or the metrics do not load them. All vectors are L2-normalised
float32, so cosine similarity equals the dot product.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Iterable

import numpy as np


def _device(device: str | None = None) -> str:
    if device:
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


class HashingEmbedder:
    """Hashed bag of words. Deterministic and dependency-free, but not semantic."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        return v

    def embed_documents(self, texts: Iterable[str], **_) -> np.ndarray:
        texts = list(texts)
        return _normalize(np.stack([self._vec(t) for t in texts])) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    # same method names as ClipEmbedder
    embed_texts = embed_documents

    def embed_images(self, paths: Iterable[str], **_) -> np.ndarray:
        return self.embed_documents([str(p) for p in paths])


def _is_hash(name: str) -> bool:
    return str(name).startswith("hash:")


class TextEmbedder:
    def __init__(self, model_name: str, query_instruction: str = "", device: str | None = None):
        self.model_name = model_name
        self.query_instruction = query_instruction or ""
        if _is_hash(model_name):
            self._impl = HashingEmbedder(int(model_name.split(":")[1]))
            self.dim = self._impl.dim
            return
        from sentence_transformers import SentenceTransformer
        self._impl = SentenceTransformer(model_name, device=_device(device))
        self.dim = self._impl.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        if isinstance(self._impl, HashingEmbedder):
            return self._impl.embed_documents(texts)
        return _normalize(self._impl.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                            show_progress_bar=show_progress, convert_to_numpy=True))

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_documents([self.query_instruction + query])[0]


def _as_array(feats) -> np.ndarray:
    """transformers 4.x returns a tensor here, 5.x can return a ModelOutput."""
    if not hasattr(feats, "float"):
        feats = getattr(feats, "pooler_output", None) if getattr(feats, "pooler_output", None) is not None else feats[0]
    return feats.float().cpu().numpy()


class ClipEmbedder:
    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        if _is_hash(model_name):
            self._hash = HashingEmbedder(int(model_name.split(":")[1]))
            self.dim = self._hash.dim
            return
        self._hash = None
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = _device(device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.dim = self.model.config.projection_dim

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if self._hash:
            return self._hash.embed_texts(texts)
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch_size):
                inp = self.processor(text=texts[i:i + batch_size], return_tensors="pt", padding=True,
                                     truncation=True, max_length=77).to(self.device)
                out.append(_as_array(self.model.get_text_features(**inp)))
        return _normalize(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)

    def embed_images(self, paths: list[str], batch_size: int = 32) -> np.ndarray:
        if self._hash:
            return self._hash.embed_images(paths)
        from PIL import Image
        out = []
        with self.torch.no_grad():
            for i in range(0, len(paths), batch_size):
                imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
                inp = self.processor(images=imgs, return_tensors="pt").to(self.device)
                out.append(_as_array(self.model.get_image_features(**inp)))
        return _normalize(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]


class Captioner:
    def __init__(self, model_name: str, device: str | None = None):
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor
        self.torch = torch
        self.device = _device(device)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()
        self.dtype = dtype

    def caption(self, paths: list[str], batch_size: int = 16, max_new_tokens: int = 40) -> list[str]:
        from PIL import Image
        caps = []
        with self.torch.no_grad():
            for i in range(0, len(paths), batch_size):
                imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
                inp = self.processor(images=imgs, return_tensors="pt").to(self.device, self.dtype)
                ids = self.model.generate(**inp, max_new_tokens=max_new_tokens, num_beams=3)
                caps += [c.strip() for c in self.processor.batch_decode(ids, skip_special_tokens=True)]
        return caps


class Reranker:
    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=_device(device), max_length=512)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self.model.predict([(query, p) for p in passages], show_progress_bar=False)]


# cached so the app and the evaluation load each model only once
@lru_cache(maxsize=4)
def get_text_embedder(model_name: str, query_instruction: str = "") -> TextEmbedder:
    return TextEmbedder(model_name, query_instruction)


@lru_cache(maxsize=4)
def get_clip(model_name: str) -> ClipEmbedder:
    return ClipEmbedder(model_name)


@lru_cache(maxsize=2)
def get_reranker(model_name: str) -> Reranker:
    return Reranker(model_name)
