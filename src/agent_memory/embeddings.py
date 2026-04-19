"""Embedding backends for agent-memory.

Two backends ship in-tree:

* ``SentenceTransformerEmbedder`` — high-quality dense embeddings. Used when
  the optional ``sentence-transformers`` dependency is installed.
* ``TfidfEmbedder`` — zero-dependency fallback built on top of numpy and the
  standard library. Good enough for small agent memory stores and keeps the
  package usable in slim environments (CI runners, serverless, edge).

Both implement the :class:`Embedder` protocol: ``embed(texts) -> np.ndarray``
where the returned matrix has shape ``(len(texts), dim)`` and rows are L2
normalized so that dot-product equals cosine similarity.
"""

from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from typing import Iterable, List, Optional, Protocol, Sequence

import numpy as np


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


class Embedder(Protocol):
    """Minimal embedder interface used by :class:`MemoryStore`."""

    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...

    def serialize(self) -> bytes: ...

    @classmethod
    def deserialize(cls, payload: bytes) -> "Embedder": ...


class SentenceTransformerEmbedder:
    """Wrapper around ``sentence-transformers`` models.

    The model is only imported lazily so that importing :mod:`agent_memory`
    does not require the heavy dependency.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised by users without the dep
            raise ImportError(
                "sentence-transformers is not installed. Install with "
                "`pip install agent-memory[st]` or fall back to TfidfEmbedder."
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"st:{model_name}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=True
        )
        return vectors.astype(np.float32)

    def serialize(self) -> bytes:
        return pickle.dumps({"model_name": self.model_name})

    @classmethod
    def deserialize(cls, payload: bytes) -> "SentenceTransformerEmbedder":
        data = pickle.loads(payload)
        return cls(model_name=data["model_name"])


class TfidfEmbedder:
    """Simple hashing TF-IDF embedder with L2-normalized output.

    The vocabulary is learned from corpus text at :meth:`fit` time. Callers
    that want a pre-fitted embedder can pass a corpus at construction time.
    Dimension defaults to 512 which is a reasonable tradeoff for small agent
    memory stores (a few thousand entries) while keeping vectors cheap to
    store in SQLite.
    """

    def __init__(
        self,
        dim: int = 512,
        corpus: Optional[Iterable[str]] = None,
    ) -> None:
        self.dim = dim
        self.name = f"tfidf:{dim}"
        # Document frequency per hashed bucket. Starts with a tiny smoothing
        # prior so the very first document still produces non-uniform IDF.
        self._df = np.ones(dim, dtype=np.float32)
        self._n_docs: int = 1
        if corpus is not None:
            self.fit(corpus)

    def _hash_token(self, token: str) -> int:
        # Deterministic, portable hash (Python's built-in hash is salted).
        h = 2166136261
        for ch in token.encode("utf-8"):
            h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
        return h % self.dim

    def fit(self, corpus: Iterable[str]) -> "TfidfEmbedder":
        for text in corpus:
            tokens = set(_tokenize(text))
            if not tokens:
                continue
            self._n_docs += 1
            for tok in tokens:
                self._df[self._hash_token(tok)] += 1.0
        return self

    def partial_fit(self, text: str) -> None:
        tokens = set(_tokenize(text))
        if not tokens:
            return
        self._n_docs += 1
        for tok in tokens:
            self._df[self._hash_token(tok)] += 1.0

    def _idf(self) -> np.ndarray:
        # Smooth IDF: log((N + 1) / (df + 1)) + 1
        return np.log((self._n_docs + 1.0) / (self._df + 1.0)).astype(np.float32) + 1.0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        idf = self._idf()
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _tokenize(text)
            if not tokens:
                continue
            counts = Counter(self._hash_token(t) for t in tokens)
            total = float(sum(counts.values()))
            for bucket, cnt in counts.items():
                tf = cnt / total
                matrix[row, bucket] = tf * idf[bucket]
        return _l2_normalize(matrix)

    def serialize(self) -> bytes:
        return pickle.dumps(
            {
                "dim": self.dim,
                "df": self._df,
                "n_docs": self._n_docs,
            }
        )

    @classmethod
    def deserialize(cls, payload: bytes) -> "TfidfEmbedder":
        data = pickle.loads(payload)
        emb = cls(dim=data["dim"])
        emb._df = data["df"].astype(np.float32)
        emb._n_docs = int(data["n_docs"])
        return emb


def default_embedder(prefer_sentence_transformers: bool = True) -> Embedder:
    """Return the best available embedder.

    Tries ``sentence-transformers`` first when ``prefer_sentence_transformers``
    is true, otherwise falls back to the TF-IDF embedder.
    """
    if prefer_sentence_transformers:
        try:
            return SentenceTransformerEmbedder()
        except ImportError:
            pass
    return TfidfEmbedder()


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a single query vector and a matrix of rows.

    Assumes both inputs are already L2 normalized — which both embedders
    guarantee — so this reduces to a dot product.
    """
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return matrix @ query.astype(np.float32)
