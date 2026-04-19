"""agent-memory: persistent memory for AI agents with vector similarity search."""

from agent_memory.embeddings import Embedder, SentenceTransformerEmbedder, TfidfEmbedder
from agent_memory.store import Memory, MemoryStore, SearchResult

__version__ = "0.1.0"

__all__ = [
    "MemoryStore",
    "Memory",
    "SearchResult",
    "Embedder",
    "SentenceTransformerEmbedder",
    "TfidfEmbedder",
]
