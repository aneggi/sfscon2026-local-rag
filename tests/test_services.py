"""Check model API failure modes and the web application's user-visible states."""

import json
from dataclasses import replace

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from local_rag.config import PROJECT_ROOT
from local_rag.ollama import Ollama


def test_embeddings_use_distinct_query_and_document_prefixes(settings):
    requests = []

    def handle(request):
        import json

        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0, 0.0]]})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        ollama = Ollama(settings, client)
        ollama.embed(["A passage"])
        ollama.embed(["A question"], query=True)
    assert requests[0]["input"] == ["search_document: A passage"]
    assert requests[1]["input"] == ["search_query: A question"]
    assert requests[0]["truncate"] is False


@pytest.mark.parametrize("vectors", [[], [[1.0, 0.0]], [[0.0, 0.0, 0.0]]])
def test_incompatible_embeddings_fail_before_database_mutation(settings, vectors):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"embeddings": vectors})
    )
    with httpx.Client(transport=transport) as client:
        with pytest.raises(ValueError, match="Invalid embeddings"):
            Ollama(settings, client).embed(["A passage"])


@pytest.mark.parametrize(
    "response",
    [
        {"done_reason": "length", "message": {"content": '{"answer": "Incomplete"}'}},
        {"done_reason": "stop", "message": {"content": "invalid JSON"}},
    ],
)
def test_incomplete_or_malformed_model_response_fails_closed(settings, response):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=response))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(ValueError):
            Ollama(settings, client).chat("Question", [])


def test_chat_without_context_allows_empty_sources_and_reads_editable_rules(settings, tmp_path):
    prompt = tmp_path / "system_prompt.txt"
    prompt.write_text("Answer directly and politely.")
    settings = replace(settings, system_prompt_file=prompt)
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "done_reason": "stop",
                "message": {"content": '{"answer": "Hello!", "source_ids": []}'},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        ollama = Ollama(settings, client)
        assert ollama.chat("Hello", [])["answer"] == "Hello!"
        prompt.write_text("Use short sentences.")
        ollama.chat("Hello again", [])
    assert "Answer directly and politely." in requests[0]["messages"][0]["content"]
    assert "Use short sentences." in requests[1]["messages"][0]["content"]
    assert requests[0]["messages"][1]["content"].endswith("User question:\nHello")
    assert requests[0]["format"]["properties"]["source_ids"]["maxItems"] == 0
    assert "enum" not in requests[0]["format"]["properties"]["source_ids"]["items"]


def test_web_app_loads_and_reports_download_and_ingestion_status(monkeypatch, settings):
    monkeypatch.setattr("local_rag.config.load_settings", lambda: settings)
    monkeypatch.setattr(
        "local_rag.download.download_documents",
        lambda settings, overwrite=False: ["Downloaded source.txt (80 bytes)."],
    )
    monkeypatch.setattr(
        "local_rag.ingest.ingest_documents",
        lambda settings: ["Indexed source.txt: 1 chunks."],
    )
    app = AppTest.from_file(PROJECT_ROOT / "app.py").run()
    assert not app.exception
    assert "Digital Sovereignty" in app.title[0].value
    app.sidebar.button[0].click().run()
    assert app.success[0].value.startswith("Downloaded")
    app.sidebar.button[1].click().run()
    assert app.success[0].value.startswith("Indexed")
    assert not app.exception
