import os
import time

import pytest

from agent_memory import MemoryStore
from agent_memory.embeddings import TfidfEmbedder


@pytest.fixture()
def store(tmp_path):
    db = tmp_path / "mem.db"
    s = MemoryStore(path=db, embedder=TfidfEmbedder(dim=256))
    yield s
    s.close()


def test_add_and_get(store):
    mem = store.add("the user prefers dark mode")
    fetched = store.get(mem.id)
    assert fetched is not None
    assert fetched.content == "the user prefers dark mode"
    assert fetched.namespace == "default"


def test_search_returns_most_similar(store):
    store.add("the user prefers dark mode in the ui")
    store.add("the user's cat is named moxie")
    store.add("the user lives in berlin")
    results = store.search("what theme does the user like?", limit=1)
    assert len(results) == 1
    assert "dark mode" in results[0].memory.content


def test_namespaces_are_isolated(store):
    store.add("alpha", namespace="agent-a")
    store.add("beta", namespace="agent-b")
    a = store.search("alpha", namespace="agent-a", limit=5)
    b = store.search("alpha", namespace="agent-b", limit=5)
    assert len(a) == 1 and a[0].memory.content == "alpha"
    assert len(b) == 1 and b[0].memory.content == "beta"
    assert set(store.namespaces()) == {"agent-a", "agent-b"}


def test_ttl_hides_expired_memories(store):
    store.add("short-lived", ttl=0.01)
    store.add("long-lived")
    time.sleep(0.05)
    listing = store.list()
    assert len(listing) == 1
    assert listing[0].content == "long-lived"
    assert store.count() == 1


def test_prune_removes_expired(store):
    store.add("short-lived", ttl=0.01)
    store.add("long-lived")
    time.sleep(0.05)
    assert store.prune() == 1
    assert store.count() == 1


def test_delete(store):
    mem = store.add("temporary thought")
    assert store.delete(mem.id) is True
    assert store.get(mem.id) is None
    assert store.delete(mem.id) is False


def test_metadata_filter(store):
    store.add("task: email Bob", metadata={"type": "task", "priority": "high"})
    store.add("task: lunch", metadata={"type": "task", "priority": "low"})
    store.add("note about email flow", metadata={"type": "note"})
    results = store.search(
        "email", metadata_filter={"type": "task", "priority": "high"}
    )
    assert len(results) == 1
    assert results[0].memory.metadata["priority"] == "high"


def test_persistence_roundtrip(tmp_path):
    db = tmp_path / "mem.db"
    s1 = MemoryStore(path=db, embedder=TfidfEmbedder(dim=128))
    mem = s1.add("remember this across restarts")
    s1.close()

    s2 = MemoryStore(path=db)  # no embedder passed — restore from disk
    fetched = s2.get(mem.id)
    assert fetched is not None
    assert fetched.content == "remember this across restarts"
    # Search still works after reopen.
    results = s2.search("restarts", limit=1)
    assert len(results) == 1
    s2.close()


def test_embedder_mismatch_raises(tmp_path):
    db = tmp_path / "mem.db"
    s1 = MemoryStore(path=db, embedder=TfidfEmbedder(dim=128))
    s1.add("hi")
    s1.close()

    with pytest.raises(ValueError):
        MemoryStore(path=db, embedder=TfidfEmbedder(dim=64))


def test_add_rejects_empty_content(store):
    with pytest.raises(ValueError):
        store.add("   ")


def test_add_many(store):
    mems = store.add_many(["alpha", "beta", "gamma"], namespace="batch")
    assert len(mems) == 3
    assert store.count(namespace="batch") == 3


def test_clear(store):
    store.add("a", namespace="x")
    store.add("b", namespace="y")
    assert store.clear(namespace="x") == 1
    assert store.count(namespace="x") == 0
    assert store.count(namespace="y") == 1
