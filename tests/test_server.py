import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent_memory import MemoryStore  # noqa: E402
from agent_memory.embeddings import TfidfEmbedder  # noqa: E402
from agent_memory.server import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    store = MemoryStore(path=tmp_path / "srv.db", embedder=TfidfEmbedder(dim=128))
    app = create_app(store)
    with TestClient(app) as c:
        yield c
    store.close()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_crud_flow(client):
    r = client.post(
        "/memories",
        json={"content": "the user prefers dark mode", "metadata": {"source": "chat"}},
    )
    assert r.status_code == 201, r.text
    mem_id = r.json()["id"]

    r = client.get(f"/memories/{mem_id}")
    assert r.status_code == 200
    assert r.json()["content"] == "the user prefers dark mode"

    r = client.get("/memories/search", params={"q": "what theme?", "limit": 3})
    assert r.status_code == 200
    results = r.json()
    assert len(results) == 1
    assert "dark mode" in results[0]["memory"]["content"]

    r = client.delete(f"/memories/{mem_id}")
    assert r.status_code == 204

    r = client.get(f"/memories/{mem_id}")
    assert r.status_code == 404


def test_namespace_isolation_over_http(client):
    client.post("/memories", json={"content": "alpha", "namespace": "ns-a"})
    client.post("/memories", json={"content": "beta", "namespace": "ns-b"})
    r = client.get("/memories/search", params={"q": "alpha", "namespace": "ns-a"})
    assert r.json()[0]["memory"]["content"] == "alpha"
    r = client.get("/memories/search", params={"q": "alpha", "namespace": "ns-b"})
    assert r.json()[0]["memory"]["content"] == "beta"


def test_stats_endpoint(client):
    client.post("/memories", json={"content": "one"})
    client.post("/memories", json={"content": "two", "namespace": "other"})
    r = client.get("/stats")
    body = r.json()
    assert body["total"] == 2
    assert set(body["namespaces"]) == {"default", "other"}
    assert body["embedder"].startswith("tfidf:")


def test_rejects_empty_content(client):
    r = client.post("/memories", json={"content": "  "})
    assert r.status_code == 400
