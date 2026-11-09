"""Small adapters for Ollama's local HTTP API; no external inference service."""

import json
import math

import httpx

from local_rag.config import Settings, load_settings


class Ollama:
    def __init__(self, settings: Settings, client: httpx.Client):
        self.settings = settings
        self.client = client

    def request(self, endpoint: str, payload: dict | None = None) -> dict:
        url = self.settings.ollama_url + endpoint
        response = self.client.get(url) if payload is None else self.client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def embedding_digest(self) -> str:
        """Model tags can move: use the downloaded weights' digest in the index identity."""
        for model in self.request("/api/tags")["models"]:
            if model["name"] == self.settings.embedding_model:
                return model["digest"]
        raise ValueError("Embedding model is missing. Run: docker compose run --rm models")

    def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        prefix = self.settings.query_prefix if query else self.settings.document_prefix
        result = self.request(
            "/api/embed",
            {
                "model": self.settings.embedding_model,
                "input": [prefix + text for text in texts],
                "truncate": False,
                "keep_alive": "5m",
            },
        )
        vectors = result["embeddings"]
        if len(vectors) != len(texts) or any(
            len(vector) != self.settings.embedding_dimensions
            or not all(math.isfinite(number) for number in vector)
            or not any(vector)
            for vector in vectors
        ):
            raise ValueError("Invalid embeddings. Check embedding_dimensions in settings.toml.")
        return vectors

    def chat(self, question: str, passages: list[dict]) -> dict:
        source_ids = [passage["passage_id"] for passage in passages]
        source_schema = {
            "type": "array",
            "maxItems": min(2, len(source_ids)),
            "items": {"type": "integer"},
        }
        if source_ids:
            source_schema["items"]["enum"] = source_ids
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "source_ids": source_schema,
            },
            "required": ["answer", "source_ids"],
            "additionalProperties": False,
        }
        system = (
            f"You are a local AI assistant using the model {self.settings.language_model}.\n\n"
            + self.settings.system_prompt_file.read_text(encoding="utf-8")
            + "\n\nReturn a nonempty answer in JSON matching the following schema. "
            "For a general answer use source_ids: []. For a document-based answer, "
            "select up to two supporting passage_id values in source_ids. "
            "Never invent a passage ID or repeat an ID. Schema: " + json.dumps(schema)
        )
        result = self.request(
            "/api/chat",
            {
                "model": self.settings.language_model,
                "stream": False,
                # Unload the larger model after each answer to keep CPU-only memory use modest.
                "keep_alive": 0,
                "format": schema,
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": (
                            "Document passages (optional context):\n"
                            + json.dumps(passages, ensure_ascii=False)
                            + "\n\nUser question:\n"
                            + question
                        ),
                    },
                ],
                "options": {
                    "temperature": 0,
                    "num_ctx": self.settings.context_tokens,
                    "num_predict": self.settings.maximum_answer_tokens,
                },
            },
        )
        if result.get("done_reason") != "stop":
            raise ValueError("The model did not finish its answer. Try a shorter question.")
        try:
            return json.loads(result["message"]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                "The model returned an unreadable response. Please try again."
            ) from error


def prepare_models(settings: Settings) -> None:
    """Only pull missing models, preserving cached weights and repeatable restarts."""
    with httpx.Client(timeout=settings.timeout_seconds, trust_env=False) as client:
        ollama = Ollama(settings, client)
        installed = {model["name"] for model in ollama.request("/api/tags")["models"]}
        for name in (settings.language_model, settings.embedding_model):
            if name in installed:
                print(f"Model already available: {name}", flush=True)
                continue
            print(f"Downloading {name}; the first startup can take several minutes.", flush=True)
            last_progress = None
            with client.stream(
                "POST",
                settings.ollama_url + "/api/pull",
                json={
                    "model": name,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if "error" in event:
                        raise RuntimeError(event["error"])
                    total = event.get("total", 0)
                    progress = (
                        event.get("status", ""),
                        int(event.get("completed", 0) / total * 10) * 10 if total else None,
                    )
                    if progress != last_progress:
                        print(f"{name}: {progress[0]} {progress[1] or 0}%", flush=True)
                        last_progress = progress
        ollama.embedding_digest()
        ollama.embed(["Check embedding dimensions before starting the web app."])


if __name__ == "__main__":
    prepare_models(load_settings())
