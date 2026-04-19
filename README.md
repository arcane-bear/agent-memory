# agent-memory

[![CI](https://github.com/arcane-bear/agent-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/arcane-bear/agent-memory/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/agent-memory.svg)](https://pypi.org/project/agent-memory/)

**Persistent memory for AI agents — vector similarity search over SQLite, no heavy dependencies.**

`agent-memory` is a drop-in memory layer for any agent framework. It gives your agents the ability to **store**, **retrieve**, and **search** memories across sessions. It runs as a standalone REST server, or embedded as a Python library — same API either way.

```python
from agent_memory import MemoryStore

store = MemoryStore("./agent.db")
store.add("the user prefers dark mode")
store.add("the user's cat is named moxie", ttl=86400)

results = store.search("what theme does the user like?")
# → [SearchResult(memory=..., score=0.62)]
```

Built by [Rapid Claw](https://rapidclaw.dev) — production infrastructure for AI agents.

## Why another memory library?

Most vector memory tools for agents pull in a heavyweight dependency — Chroma, Pinecone, Weaviate, FAISS. For the **small-to-medium memory stores** that most agents actually need (thousands to low hundreds of thousands of entries), that's overkill.

`agent-memory` is deliberately small:

- **SQLite + numpy** for storage and search — zero external services.
- **Optional embeddings:** uses [`sentence-transformers`](https://www.sbert.net/) when installed, falls back to a hashing TF-IDF embedder with **zero extra dependencies** so the package works anywhere Python does.
- **One file, one API.** Use it as a library, a REST server, or a CLI — they share the same `MemoryStore` underneath.
- **Namespaces + TTL** built in, so multi-agent and multi-session setups don't need a second library.

## Install

```bash
# Minimal — library-only, TF-IDF fallback embedder
pip install agent-memory

# With the REST server
pip install "agent-memory[server]"

# With high-quality sentence-transformer embeddings
pip install "agent-memory[st]"

# Everything (dev + server + embeddings)
pip install "agent-memory[dev,server,st]"
```

## Library usage

```python
from agent_memory import MemoryStore

# Persistent on disk — or pass ":memory:" for an ephemeral store.
store = MemoryStore("./memory.db", namespace="agent-1")

# Write.
mem = store.add(
    "the user wants emails summarized, not listed verbatim",
    metadata={"source": "preferences"},
    ttl=30 * 86400,  # expires in 30 days
)

# Search by similarity.
hits = store.search("how should I present emails?", limit=3)
for h in hits:
    print(h.score, h.memory.content)

# Scope by namespace (per-agent or per-session memory).
store.add("Bob asked about refunds", namespace="session-42")
store.search("refunds", namespace="session-42")

# Metadata filters.
store.search("email", metadata_filter={"source": "preferences"})

# Housekeeping.
store.prune()            # drop expired rows
store.delete(mem.id)     # drop one by id
store.clear("session-42")  # drop a whole namespace
```

### Picking an embedder

```python
from agent_memory import MemoryStore, SentenceTransformerEmbedder, TfidfEmbedder

# Best quality when sentence-transformers is installed:
store = MemoryStore("./m.db", embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"))

# Zero-dependency fallback:
store = MemoryStore("./m.db", embedder=TfidfEmbedder(dim=512))

# Default: sentence-transformers if available, TF-IDF otherwise.
store = MemoryStore("./m.db")
```

The embedder identity is stored in the database — reopening an existing store picks up the same embedder automatically. Mixing embedders against the same file raises a clear error.

## REST server

```bash
agent-memory serve --host 0.0.0.0 --port 8000 --db ./memory.db
# or, directly:
uvicorn agent_memory.server:app --reload
```

### Endpoints

| Method   | Path                      | Description                                   |
| -------- | ------------------------- | --------------------------------------------- |
| `GET`    | `/health`                 | Liveness check.                               |
| `GET`    | `/stats`                  | Total count, namespaces, embedder name, dim.  |
| `POST`   | `/memories`               | Create a memory. Body: `{content, namespace?, metadata?, ttl?, id?}`. |
| `GET`    | `/memories`               | List. Query: `namespace`, `limit`, `offset`.  |
| `GET`    | `/memories/search`        | Similarity search. Query: `q`, `namespace`, `limit`, `min_score`. |
| `GET`    | `/memories/{id}`          | Fetch a memory by id.                         |
| `DELETE` | `/memories/{id}`          | Delete a memory.                              |
| `POST`   | `/prune`                  | Delete expired memories.                      |

### Example

```bash
curl -X POST http://localhost:8000/memories \
  -H 'content-type: application/json' \
  -d '{"content": "the user is allergic to shellfish", "namespace": "user-42"}'

curl "http://localhost:8000/memories/search?q=what+can+the+user+eat&namespace=user-42&limit=3"
```

Response:

```json
[
  {
    "memory": {
      "id": "a7c3...",
      "namespace": "user-42",
      "content": "the user is allergic to shellfish",
      "metadata": {},
      "created_at": 1744800000.0,
      "expires_at": null
    },
    "score": 0.41
  }
]
```

## CLI

```bash
agent-memory add "the user's favorite editor is Helix" --namespace user-42
agent-memory search "what editor does the user prefer?" -n user-42 -k 3
agent-memory list --namespace user-42
agent-memory stats
agent-memory prune
agent-memory serve --port 8000
```

## Integrating with agent frameworks

`agent-memory` is framework-agnostic. The usual pattern is:

```python
def on_user_turn(user_input: str, session_id: str) -> str:
    # 1. Pull relevant memories before the LLM call.
    memories = store.search(user_input, namespace=session_id, limit=5)
    context = "\n".join(f"- {m.memory.content}" for m in memories)

    # 2. Call your LLM with the memory-augmented prompt.
    reply = llm.complete(system=context, user=user_input)

    # 3. Persist anything worth remembering.
    if should_remember(reply):
        store.add(extract_fact(reply), namespace=session_id)

    return reply
```

See [`examples/agent_loop.py`](examples/agent_loop.py) for a runnable version.

## Design notes

- **Storage:** Each memory is one SQLite row; the embedding lives in a `BLOB` column as raw `float32`. Search is a single `SELECT` followed by a numpy matmul. This is dramatically faster than an ANN index for the sizes most agents need (< ~100k rows) and keeps install size small.
- **Namespaces** are an indexed string column. Use them for per-user, per-session, or per-agent partitioning — queries always scope to one namespace.
- **TTL** is stored as an absolute `expires_at` timestamp. Expired rows are hidden from reads and cleaned up lazily by `prune()` (or the `POST /prune` endpoint).
- **TF-IDF fallback** uses a fixed-size hashing vectorizer with smoothed IDF updated on each write. Portable hash, L2-normalized output, drop-in for cosine similarity.

## When NOT to use this

- You need **ANN** over millions of vectors — use FAISS or a managed vector DB.
- You need **clustering, hybrid search, or reranking** out of the box — use a fuller-featured stack.
- You need **cross-process writers** at high throughput — SQLite works fine for one writer and many readers, but for write-heavy concurrent workloads prefer a dedicated store.

For a single agent, or a handful of agents sharing memory on one box, this library is the boring, reliable choice.

## Development

```bash
git clone https://github.com/arcane-bear/agent-memory
cd agent-memory
pip install -e ".[dev,server]"
pytest
ruff check src tests
```

## Learn More

Learn more about [AI agent memory management](https://rapidclaw.dev/blog/ai-agent-memory-state-management) on the Rapid Claw blog.

Explore [rapidclaw.dev](https://rapidclaw.dev) for more open-source agent tooling.

## License

[MIT](LICENSE).
