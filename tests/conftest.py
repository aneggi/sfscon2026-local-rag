"""Small in-memory substitutes keep the pipeline tests fast and fully offline."""

from dataclasses import replace

import pytest

from local_rag.config import load_settings


@pytest.fixture
def settings(tmp_path):
    sources = tmp_path / "sources.toml"
    sources.write_text("documents = []\n", encoding="utf-8")
    documents = tmp_path / "docs"
    documents.mkdir()
    return replace(
        load_settings(),
        documents_dir=documents,
        state_dir=tmp_path / "state",
        sources_file=sources,
        embedding_dimensions=3,
    )


class MemoryOllama:
    embedded_texts = []
    fail_embeddings = False
    generated = {}
    chat_calls = 0

    def __init__(self, settings, client):
        pass

    def embedding_digest(self):
        return "test-model-digest"

    def embed(self, texts, *, query=False):
        if self.fail_embeddings:
            raise ValueError("Simulated embedding failure")
        self.embedded_texts.extend(texts)
        return [[1.0, 0.0, 0.0] for text in texts]

    def chat(self, question, passages):
        type(self).chat_calls += 1
        return self.generated


class MemoryStore:
    collections = {}
    no_hits = False
    fail_after_upsert = False

    def __init__(self, settings, client, collection):
        self.collection = collection

    def exists(self):
        return self.collection in self.collections

    def ensure_collection(self):
        self.collections.setdefault(self.collection, {})

    def count(self, filename, revision):
        return sum(
            point["payload"]["filename"] == filename and point["payload"]["revision"] == revision
            for point in self.collections[self.collection].values()
        )

    def upsert(self, points):
        self.collections[self.collection].update({point["id"]: point for point in points})
        if self.fail_after_upsert:
            raise ValueError("Simulated interruption after a database write")

    def delete(self, filename, *, keep_revision=None):
        points = self.collections[self.collection]
        for point_id, point in list(points.items()):
            payload = point["payload"]
            if payload["filename"] == filename and payload["revision"] != keep_revision:
                del points[point_id]

    def search(self, vector, manifest):
        if self.no_hits:
            return []
        return [
            {"payload": point["payload"], "score": 0.9}
            for point in self.collections[self.collection].values()
            if point["payload"]["revision"]
            == manifest.get(point["payload"]["filename"], {}).get("revision")
        ][:4]


@pytest.fixture
def memory_services(monkeypatch):
    from local_rag import chat, ingest

    MemoryStore.collections = {}
    MemoryStore.no_hits = False
    MemoryStore.fail_after_upsert = False
    MemoryOllama.embedded_texts = []
    MemoryOllama.fail_embeddings = False
    MemoryOllama.generated = {"answer": "A general answer.", "source_ids": []}
    MemoryOllama.chat_calls = 0
    for module in (ingest, chat):
        monkeypatch.setattr(module, "Ollama", MemoryOllama)
        monkeypatch.setattr(module, "VectorStore", MemoryStore)
    return MemoryOllama, MemoryStore
