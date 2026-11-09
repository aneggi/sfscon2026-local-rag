"""Verify updates, recovery, general answers, and document citations without model weights."""

from dataclasses import replace

import pytest

from local_rag.chat import answer_question, validate_answer
from local_rag.ingest import ingest_documents
from local_rag.state import read_json

TEXT = "An open license must allow use, redistribution, and modification for any purpose."


def test_unchanged_documents_do_not_generate_embeddings_again(settings, memory_services):
    ollama, store = memory_services
    (settings.documents_dir / "license.txt").write_text(TEXT)
    assert ingest_documents(settings) == ["Indexed license.txt: 1 chunks."]
    assert ingest_documents(settings) == ["Skipped license.txt: unchanged (1 chunks)."]
    assert ollama.embedded_texts == [TEXT]
    assert len(next(iter(store.collections.values()))) == 1


def test_changed_document_replaces_old_chunks_and_deleted_file_is_removed(
    settings, memory_services
):
    _, store = memory_services
    path = settings.documents_dir / "license.txt"
    path.write_text(TEXT * 30)
    ingest_documents(settings)
    assert len(next(iter(store.collections.values()))) > 1
    path.write_text(TEXT)
    ingest_documents(settings)
    points = next(iter(store.collections.values()))
    assert len(points) == 1
    assert next(iter(points.values()))["payload"]["text"] == TEXT
    path.unlink()
    assert ingest_documents(settings) == ["Removed license.txt from the index."]
    assert not next(iter(store.collections.values()))


def test_embedding_failure_keeps_previous_document_usable(settings, memory_services):
    ollama, store = memory_services
    path = settings.documents_dir / "license.txt"
    path.write_text(TEXT)
    ingest_documents(settings)
    path.write_text("A replacement document with new information.")
    ollama.fail_embeddings = True
    assert ingest_documents(settings)[0].startswith("ERROR")
    points = next(iter(store.collections.values()))
    assert len(points) == 1
    assert next(iter(points.values()))["payload"]["text"] == TEXT
    ollama.fail_embeddings = False
    ingest_documents(settings)
    assert next(iter(points.values()))["payload"]["text"] != TEXT


def test_interrupted_write_is_hidden_and_retry_does_not_duplicate_chunks(settings, memory_services):
    ollama, store = memory_services
    path = settings.documents_dir / "license.txt"
    path.write_text(TEXT)
    ingest_documents(settings)
    path.write_text("New information must not become searchable before its commit.")
    store.fail_after_upsert = True
    assert ingest_documents(settings)[0].startswith("ERROR")
    # Database contains both versions, but only the committed one is retrieved.
    points = next(iter(store.collections.values()))
    assert len(points) == 2
    answer = answer_question(settings, "What permissions are required?")
    assert [passage["text"] for passage in answer.retrieved_passages] == [TEXT]
    store.fail_after_upsert = False
    ingest_documents(settings)
    assert len(points) == 1
    assert ingest_documents(settings)[0].startswith("Skipped")


def test_lost_database_is_rebuilt_even_when_manifest_is_unchanged(settings, memory_services):
    ollama, store = memory_services
    (settings.documents_dir / "license.txt").write_text(TEXT)
    ingest_documents(settings)
    store.collections.clear()
    assert ingest_documents(settings) == ["Indexed license.txt: 1 chunks."]
    assert len(ollama.embedded_texts) == 2


def test_changed_source_metadata_is_reindexed(settings, memory_services):
    _, store = memory_services
    (settings.documents_dir / "license.txt").write_text(TEXT)
    ingest_documents(settings)
    settings.sources_file.write_text(
        '[[documents]]\nfilename = "license.txt"\nurl = "https://example.org/license.txt"\n'
    )
    assert ingest_documents(settings) == ["Indexed license.txt: 1 chunks."]
    point = next(iter(next(iter(store.collections.values())).values()))
    assert point["payload"]["source_url"] == "https://example.org/license.txt"


def test_changed_chunk_settings_create_a_separate_index(settings, memory_services):
    _, store = memory_services
    (settings.documents_dir / "license.txt").write_text(TEXT)
    ingest_documents(settings)
    ingest_documents(replace(settings, chunk_size=500))
    assert len(store.collections) == 2


def test_force_reembeds_without_duplicates(settings, memory_services):
    ollama, store = memory_services
    (settings.documents_dir / "license.txt").write_text(TEXT)
    ingest_documents(settings)
    ingest_documents(settings, force=True)
    assert len(ollama.embedded_texts) == 2
    assert len(next(iter(store.collections.values()))) == 1


def test_invalid_pdf_does_not_stop_other_documents(settings, memory_services):
    (settings.documents_dir / "broken.pdf").write_bytes(b"not a PDF")
    (settings.documents_dir / "license.txt").write_text(TEXT)
    messages = ingest_documents(settings)
    assert messages[0].startswith("ERROR broken.pdf")
    assert messages[1] == "Indexed license.txt: 1 chunks."


@pytest.mark.parametrize("question", ["Are you an AI?", "Explain what a vector is."])
def test_general_questions_work_without_an_index(settings, memory_services, question):
    ollama, _ = memory_services
    ollama.generated = {"answer": "I can answer general questions.", "source_ids": []}
    answer = answer_question(settings, question)
    assert answer.text == ollama.generated["answer"]
    assert not answer.sources
    assert not answer.retrieved_passages
    assert not ollama.embedded_texts
    assert ollama.chat_calls == 1


def test_no_document_matches_still_calls_language_model(settings, memory_services):
    ollama, store = memory_services
    (settings.documents_dir / "license.txt").write_text(TEXT)
    ingest_documents(settings)
    store.no_hits = True
    answer = answer_question(settings, "Explain what a vector is.")
    assert answer.text == ollama.generated["answer"]
    assert not answer.sources
    assert ollama.chat_calls == 1


@pytest.mark.parametrize("question", ["Are you an AI?", "Hello!", "Who are you?"])
def test_greetings_and_identity_use_model_without_unrelated_documents(
    settings, memory_services, question
):
    ollama, _ = memory_services
    (settings.documents_dir / "license.txt").write_text(TEXT)
    ingest_documents(settings)
    answer = answer_question(settings, question)
    assert answer.text == ollama.generated["answer"]
    assert not answer.sources
    assert not answer.retrieved_passages
    assert ollama.embedded_texts == [TEXT]
    assert ollama.chat_calls == 1


def test_answer_includes_verified_source_and_excerpt(settings, memory_services):
    ollama, _ = memory_services
    (settings.documents_dir / "license.txt").write_text(TEXT)
    ingest_documents(settings)
    ollama.generated = {
        "answer": "The license must allow use for any purpose.",
        "source_ids": [1],
    }
    answer = answer_question(settings, "What permissions are required?")
    assert answer.sources[0]["filename"] == "license.txt"
    assert answer.sources[0]["text"] == TEXT
    assert answer.text == ollama.generated["answer"]


def test_malformed_answers_and_invented_citations_raise_clear_errors():
    passages = [{"passage_id": 1, "text": TEXT, "filename": "license.txt"}]
    for generated in (
        None,
        [],
        {},
        {"answer": "", "source_ids": []},
        {"answer": "Invalid source list", "source_ids": None},
        {"answer": "Invented source", "source_ids": [99]},
        {"answer": "Mixed sources", "source_ids": [1, 99]},
        {"answer": "Invalid ID", "source_ids": [True]},
        {"answer": "Wrong ID type", "source_ids": ["1"]},
    ):
        with pytest.raises(ValueError):
            validate_answer(generated, passages)


@pytest.mark.parametrize("passages", [[], [{"passage_id": 1, "text": TEXT}]])
def test_general_answers_do_not_require_document_citations(passages):
    answer = validate_answer({"answer": "Yes, I am an AI assistant.", "source_ids": []}, passages)
    assert answer.text == "Yes, I am an AI assistant."
    assert not answer.sources


def test_source_text_comes_from_retrieval_and_duplicate_references_are_collapsed():
    passages = [
        {"passage_id": 1, "text": "An unrelated passage.", "filename": "unrelated.txt"},
        {"passage_id": 2, "text": TEXT, "filename": "license.txt"},
    ]
    answer = validate_answer(
        {
            "answer": "Use for any purpose is required.",
            "source_ids": [2, 2],
            "filename": "invented.txt",
            "text": "An invented quotation.",
        },
        passages,
    )
    assert len(answer.sources) == 1
    assert answer.sources[0]["filename"] == "license.txt"
    assert answer.sources[0]["text"] == TEXT


def test_manifest_is_persisted_as_document_revision_and_chunk_count(settings, memory_services):
    (settings.documents_dir / "license.txt").write_text(TEXT)
    ingest_documents(settings)
    manifest = read_json(next(settings.state_dir.glob("documents_*.json")))
    assert manifest["license.txt"]["chunks"] == 1
    assert len(manifest["license.txt"]["revision"]) == 64
