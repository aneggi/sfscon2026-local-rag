"""Optional wire-level check against a dedicated, real Qdrant instance."""

import os
from dataclasses import replace
from uuid import uuid4

import httpx
import pytest

from local_rag import chat, ingest
from local_rag.bootstrap import wait_for_service
from local_rag.state import read_json
from local_rag.vector_store import VectorStore, collection_name


@pytest.mark.skipif(
    not os.getenv("QDRANT_TEST_URL"), reason="Set QDRANT_TEST_URL for integration test"
)
def test_real_qdrant_ingest_query_update_delete_and_recovery(settings, monkeypatch):
    digest = str(uuid4())
    settings = replace(settings, qdrant_url=os.environ["QDRANT_TEST_URL"])

    class TestOllama:
        def __init__(self, settings, client):
            pass

        def embedding_digest(self):
            return digest

        def embed(self, texts, *, query=False):
            return [[1.0, 0.0, 0.0] for text in texts]

        def chat(self, question, passages):
            return {
                "answer": "The document describes open data.",
                "source_ids": [passages[0]["passage_id"]] if passages else [],
            }

    for module in (chat, ingest):
        monkeypatch.setattr(module, "Ollama", TestOllama)
    collection = collection_name(settings, digest)
    with httpx.Client(timeout=10, trust_env=False) as client:
        wait_for_service(client, settings.qdrant_url + "/readyz")
        store = VectorStore(settings, client, collection)
        try:
            path = settings.documents_dir / "open-data.txt"
            path.write_text("An open data document describes reuse and redistribution. " * 60)
            assert ingest.ingest_documents(settings)[0].startswith("Indexed")
            assert ingest.ingest_documents(settings)[0].startswith("Skipped")
            answer = chat.answer_question(settings, "What does the document describe?")
            assert answer.sources[0]["filename"] == "open-data.txt"
            old_revision = answer.sources[0]["revision"]
            path.write_text("An updated open data document allows commercial redistribution.")
            ingest.ingest_documents(settings)
            assert store.count("open-data.txt", old_revision) == 0
            answer = chat.answer_question(settings, "What does the document describe?")
            assert "updated" in answer.sources[0]["text"]
            # A missing collection must be repaired despite the existing local manifest.
            client.delete(store.url).raise_for_status()
            assert ingest.ingest_documents(settings)[0].startswith("Indexed")
            path.unlink()
            assert ingest.ingest_documents(settings)[0].startswith("Removed")
            assert not read_json(settings.state_dir / f"{collection}.json")
            assert not chat.answer_question(settings, "Any data?").sources
        finally:
            client.delete(store.url).raise_for_status()
