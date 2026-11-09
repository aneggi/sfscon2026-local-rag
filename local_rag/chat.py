"""Answer naturally, using retrieved document context when it is available."""

from dataclasses import dataclass, field

import httpx

from local_rag.config import Settings
from local_rag.ollama import Ollama
from local_rag.state import pipeline_lock, read_json
from local_rag.vector_store import VectorStore, collection_name


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)
    retrieved_passages: list[dict] = field(default_factory=list)


def can_skip_document_search(question: str) -> bool:
    """Greetings and self-description do not benefit from unrelated PDF context."""
    normalized = " ".join(question.casefold().split()).rstrip(" .?!")
    return normalized in {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "how are you",
        "are you an ai",
        "who are you",
        "what are you",
        "what can you do",
        "what model are you using",
    }


def validate_answer(generated: dict, passages: list[dict]) -> Answer:
    """Resolve selected IDs to original passages; never generate source text or filenames."""
    if not isinstance(generated, dict):
        raise ValueError("The model returned an invalid answer. Please try again.")
    text, source_ids = generated.get("answer"), generated.get("source_ids")
    if not isinstance(text, str) or not text.strip() or not isinstance(source_ids, list):
        raise ValueError("The model returned an invalid answer. Please try again.")
    by_id = {passage["passage_id"]: passage for passage in passages}
    if any(type(source_id) is not int or source_id not in by_id for source_id in source_ids):
        raise ValueError("The model cited an unknown document passage. Please try again.")
    sources = [by_id[source_id] for source_id in dict.fromkeys(source_ids)]
    return Answer(text.strip(), sources, passages)


def answer_question(settings: Settings, question: str) -> Answer:
    if not question.strip():
        raise ValueError("Please enter a question.")
    if len(question) > 1000:
        raise ValueError("Keep the question under 1,000 characters for this small local model.")
    with (
        pipeline_lock(settings.state_dir, shared=True),
        httpx.Client(timeout=settings.timeout_seconds, trust_env=False) as client,
    ):
        ollama = Ollama(settings, client)
        hits = []
        # An empty library must not prevent ordinary conversation.
        if not can_skip_document_search(question) and any(
            settings.state_dir.glob("documents_*.json")
        ):
            collection = collection_name(settings, ollama.embedding_digest())
            manifest = read_json(settings.state_dir / f"{collection}.json")
            store = VectorStore(settings, client, collection)
            if manifest and store.exists():
                vector = ollama.embed([question], query=True)[0]
                hits = store.search(vector, manifest)
        passages = [
            {"passage_id": number, "score": hit["score"], **hit["payload"]}
            for number, hit in enumerate(hits, 1)
        ]
        # Reserve the answer budget and space for instructions, JSON, and the question.
        # A UTF-8 byte budget is deliberately conservative: at most one token per byte.
        prompt_passages = []
        budget = (
            settings.context_tokens
            - settings.maximum_answer_tokens
            - 1600
            - len(question.encode("utf-8"))
        )
        for passage in passages:
            cost = len(passage["text"].encode("utf-8")) + 80
            if cost > budget:
                break
            prompt_passages.append({"passage_id": passage["passage_id"], "text": passage["text"]})
            budget -= cost
        sent_ids = {passage["passage_id"] for passage in prompt_passages}
        sent_passages = [passage for passage in passages if passage["passage_id"] in sent_ids]
        generated = ollama.chat(question, prompt_passages)
        return validate_answer(generated, sent_passages)
