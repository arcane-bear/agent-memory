"""Core :class:`MemoryStore` — SQLite-backed vector memory for agents.

Design notes
------------

* Vectors are stored in the same SQLite row as the memory text, as raw
  float32 bytes. For small-to-medium agent memories (up to ~100k rows) a
  simple in-memory cosine scan over a numpy matrix is faster and simpler
  than any ANN index, and keeps the package dependency-free.

* Namespaces let callers partition memory per-agent, per-session, or
  per-user without running multiple stores. The namespace is indexed and
  every query scopes to it.

* TTLs are stored as absolute expiry timestamps. Expired rows are filtered
  out of search results and cleaned up lazily via :meth:`MemoryStore.prune`.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from agent_memory.embeddings import (
    Embedder,
    TfidfEmbedder,
    cosine_similarity,
    default_embedder,
)

DEFAULT_NAMESPACE = "default"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    embedding BLOB NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL
);
CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
CREATE INDEX IF NOT EXISTS idx_memories_expires_at ON memories(expires_at);

CREATE TABLE IF NOT EXISTS embedder_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    name TEXT NOT NULL,
    dim INTEGER NOT NULL,
    payload BLOB NOT NULL
);
"""


@dataclass
class Memory:
    """A stored memory entry."""

    id: str
    namespace: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    expires_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    """A memory plus the similarity score from a search query."""

    memory: Memory
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"memory": self.memory.to_dict(), "score": self.score}


class MemoryStore:
    """Persistent vector memory store.

    Parameters
    ----------
    path:
        Path to the SQLite file. Use ``":memory:"`` for an ephemeral store
        (useful in tests and notebooks).
    embedder:
        An :class:`Embedder` instance. When omitted, :func:`default_embedder`
        picks sentence-transformers if available, otherwise a TF-IDF
        fallback.
    namespace:
        Default namespace used by :meth:`add` and :meth:`search` when the
        caller does not pass one.
    """

    def __init__(
        self,
        path: Union[str, Path] = "agent_memory.db",
        embedder: Optional[Embedder] = None,
        namespace: str = DEFAULT_NAMESPACE,
    ) -> None:
        self.path = str(path)
        self.default_namespace = namespace
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        stored = self._load_embedder_state()
        if embedder is not None:
            self.embedder = embedder
            if stored and stored[0] != embedder.name:
                raise ValueError(
                    f"Existing store was built with embedder '{stored[0]}', "
                    f"refusing to mix with '{embedder.name}'. Pass the same "
                    "embedder or start a fresh store."
                )
        elif stored is not None:
            name, _dim, payload = stored
            if name.startswith("tfidf:"):
                self.embedder = TfidfEmbedder.deserialize(payload)
            else:
                # Lazy import to avoid hard dependency.
                from agent_memory.embeddings import SentenceTransformerEmbedder

                self.embedder = SentenceTransformerEmbedder.deserialize(payload)
        else:
            self.embedder = default_embedder()

        self._save_embedder_state()

    # ------------------------------------------------------------------ utils

    def _now(self) -> float:
        return time.time()

    def _load_embedder_state(self):
        row = self._conn.execute(
            "SELECT name, dim, payload FROM embedder_state WHERE id = 1"
        ).fetchone()
        return row

    def _save_embedder_state(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO embedder_state(id, name, dim, payload) VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET name=excluded.name, dim=excluded.dim, payload=excluded.payload
                """,
                (self.embedder.name, self.embedder.dim, self.embedder.serialize()),
            )
            self._conn.commit()

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        return Memory(
            id=row[0],
            namespace=row[1],
            content=row[2],
            metadata=json.loads(row[3]) if row[3] else {},
            created_at=row[5],
            expires_at=row[6],
        )

    # ---------------------------------------------------------------- writes

    def add(
        self,
        content: str,
        *,
        namespace: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[float] = None,
        id: Optional[str] = None,
    ) -> Memory:
        """Store a memory. Returns the :class:`Memory` row that was written.

        ``ttl`` is seconds-until-expiry. Rows past their expiry are hidden
        from :meth:`search` and purged by :meth:`prune`.
        """
        if not content or not content.strip():
            raise ValueError("content must be a non-empty string")

        ns = namespace or self.default_namespace
        mem_id = id or uuid.uuid4().hex
        created_at = self._now()
        expires_at = created_at + ttl if ttl is not None else None
        meta_json = json.dumps(metadata or {})

        # Update TF-IDF vocabulary before embedding so new terms get weight.
        if isinstance(self.embedder, TfidfEmbedder):
            self.embedder.partial_fit(content)

        vector = self.embedder.embed([content])[0].astype(np.float32)

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memories(id, namespace, content, metadata, embedding, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mem_id,
                    ns,
                    content,
                    meta_json,
                    vector.tobytes(),
                    created_at,
                    expires_at,
                ),
            )
            self._conn.commit()
            if isinstance(self.embedder, TfidfEmbedder):
                self._save_embedder_state()

        return Memory(
            id=mem_id,
            namespace=ns,
            content=content,
            metadata=metadata or {},
            created_at=created_at,
            expires_at=expires_at,
        )

    def add_many(
        self,
        contents: Sequence[str],
        *,
        namespace: Optional[str] = None,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ttl: Optional[float] = None,
    ) -> List[Memory]:
        """Batch-insert memories. Embeddings are computed in one call."""
        if not contents:
            return []
        if metadatas is not None and len(metadatas) != len(contents):
            raise ValueError("metadatas must match contents in length")

        ns = namespace or self.default_namespace
        now = self._now()
        expires_at = now + ttl if ttl is not None else None

        if isinstance(self.embedder, TfidfEmbedder):
            for c in contents:
                self.embedder.partial_fit(c)

        vectors = self.embedder.embed(list(contents)).astype(np.float32)
        memories: List[Memory] = []
        rows = []
        for idx, content in enumerate(contents):
            mem_id = uuid.uuid4().hex
            md = metadatas[idx] if metadatas else {}
            rows.append(
                (
                    mem_id,
                    ns,
                    content,
                    json.dumps(md),
                    vectors[idx].tobytes(),
                    now,
                    expires_at,
                )
            )
            memories.append(
                Memory(
                    id=mem_id,
                    namespace=ns,
                    content=content,
                    metadata=md,
                    created_at=now,
                    expires_at=expires_at,
                )
            )

        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO memories(id, namespace, content, metadata, embedding, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()
            if isinstance(self.embedder, TfidfEmbedder):
                self._save_embedder_state()

        return memories

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by id. Returns whether a row was removed."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def clear(self, namespace: Optional[str] = None) -> int:
        """Clear a namespace (or the whole store when ``namespace`` is None)."""
        with self._lock:
            if namespace is None:
                cur = self._conn.execute("DELETE FROM memories")
            else:
                cur = self._conn.execute(
                    "DELETE FROM memories WHERE namespace = ?", (namespace,)
                )
            self._conn.commit()
            return cur.rowcount

    def prune(self) -> int:
        """Remove expired memories. Returns the count deleted."""
        now = self._now()
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
            self._conn.commit()
            return cur.rowcount

    # ----------------------------------------------------------------- reads

    def get(self, memory_id: str) -> Optional[Memory]:
        row = self._conn.execute(
            """
            SELECT id, namespace, content, metadata, embedding, created_at, expires_at
            FROM memories WHERE id = ?
            """,
            (memory_id,),
        ).fetchone()
        if not row:
            return None
        if row[6] is not None and row[6] <= self._now():
            return None
        return self._row_to_memory(row)

    def list(
        self,
        namespace: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Memory]:
        ns = namespace or self.default_namespace
        now = self._now()
        rows = self._conn.execute(
            """
            SELECT id, namespace, content, metadata, embedding, created_at, expires_at
            FROM memories
            WHERE namespace = ? AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (ns, now, limit, offset),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def count(self, namespace: Optional[str] = None) -> int:
        now = self._now()
        if namespace is None:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE expires_at IS NULL OR expires_at > ?",
                (now,),
            ).fetchone()
        else:
            row = self._conn.execute(
                """
                SELECT COUNT(*) FROM memories
                WHERE namespace = ? AND (expires_at IS NULL OR expires_at > ?)
                """,
                (namespace, now),
            ).fetchone()
        return int(row[0])

    def namespaces(self) -> List[str]:
        now = self._now()
        rows = self._conn.execute(
            """
            SELECT DISTINCT namespace FROM memories
            WHERE expires_at IS NULL OR expires_at > ?
            ORDER BY namespace
            """,
            (now,),
        ).fetchall()
        return [r[0] for r in rows]

    def search(
        self,
        query: str,
        *,
        namespace: Optional[str] = None,
        limit: int = 5,
        min_score: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Return the top ``limit`` memories most similar to ``query``.

        ``metadata_filter`` is a simple equality filter: every key must match
        exactly. For richer filtering, post-process the returned results.
        """
        if not query or not query.strip():
            return []
        ns = namespace or self.default_namespace
        now = self._now()
        rows = self._conn.execute(
            """
            SELECT id, namespace, content, metadata, embedding, created_at, expires_at
            FROM memories
            WHERE namespace = ? AND (expires_at IS NULL OR expires_at > ?)
            """,
            (ns, now),
        ).fetchall()
        if not rows:
            return []

        if metadata_filter:
            filtered = []
            for r in rows:
                md = json.loads(r[3]) if r[3] else {}
                if all(md.get(k) == v for k, v in metadata_filter.items()):
                    filtered.append(r)
            rows = filtered
            if not rows:
                return []

        query_vec = self.embedder.embed([query])[0].astype(np.float32)
        matrix = np.stack(
            [np.frombuffer(r[4], dtype=np.float32) for r in rows]
        )
        scores = cosine_similarity(query_vec, matrix)

        # argpartition is O(n) for top-k; fall back to argsort for tiny lists.
        k = min(limit, len(rows))
        if k >= len(rows):
            order = np.argsort(-scores)
        else:
            top = np.argpartition(-scores, k - 1)[:k]
            order = top[np.argsort(-scores[top])]

        results: List[SearchResult] = []
        for i in order[:limit]:
            score = float(scores[i])
            if score < min_score:
                continue
            results.append(SearchResult(memory=self._row_to_memory(rows[i]), score=score))
        return results

    # ----------------------------------------------------------- lifecycle

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
