import numpy as np

from agent_memory.embeddings import TfidfEmbedder, cosine_similarity


def test_tfidf_shape_and_normalization():
    emb = TfidfEmbedder(dim=128)
    emb.fit(["hello world", "goodbye world"])
    vectors = emb.embed(["hello world", "unrelated text"])
    assert vectors.shape == (2, 128)
    norms = np.linalg.norm(vectors, axis=1)
    # Non-empty rows are L2 normalized, empty rows stay 0.
    for i, row in enumerate(vectors):
        if row.sum() != 0:
            np.testing.assert_allclose(norms[i], 1.0, atol=1e-5)


def test_tfidf_similarity_ranking():
    emb = TfidfEmbedder(dim=256)
    corpus = [
        "the user prefers dark mode in the ui",
        "the user wants imperial units for weather",
        "the cat is named moxie",
    ]
    emb.fit(corpus)
    vectors = emb.embed(corpus)
    query = emb.embed(["what theme does the user like"])[0]
    scores = cosine_similarity(query, vectors)
    # The dark-mode memory should rank first.
    assert int(np.argmax(scores)) == 0


def test_tfidf_serialize_roundtrip():
    emb = TfidfEmbedder(dim=64)
    emb.fit(["alpha beta", "beta gamma", "gamma delta"])
    payload = emb.serialize()
    restored = TfidfEmbedder.deserialize(payload)
    np.testing.assert_array_equal(emb._df, restored._df)
    assert emb._n_docs == restored._n_docs
    # Same text should produce identical vectors across instances.
    np.testing.assert_allclose(emb.embed(["alpha"]), restored.embed(["alpha"]))


def test_tfidf_empty_inputs():
    emb = TfidfEmbedder(dim=32)
    assert emb.embed([]).shape == (0, 32)
    # An all-punctuation string has no tokens, so we get a zero row.
    row = emb.embed(["!!!"])[0]
    assert np.all(row == 0)
