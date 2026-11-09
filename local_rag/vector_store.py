"""Qdrant stores vectors and their original passages together as points."""

import hashlib
import json

import httpx

from local_rag.config import Settings

# Increment this when text extraction or payload semantics change.
INDEX_FORMAT_VERSION = 1


def collection_name(settings: Settings, embedding_digest: str) -> str:
    identity = {
        "format": INDEX_FORMAT_VERSION,
        "embedding_model": settings.embedding_model,
        "embedding_digest": embedding_digest,
        "dimensions": settings.embedding_dimensions,
        "document_prefix": settings.document_prefix,
        "query_prefix": settings.query_prefix,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return "documents_" + fingerprint


def document_filter(filename: str, revision: str | None = None) -> dict:
    conditions = [{"key": "filename", "match": {"value": filename}}]
    if revision is not None:
        conditions.append({"key": "revision", "match": {"value": revision}})
    return {"must": conditions}


class VectorStore:
    def __init__(self, settings: Settings, client: httpx.Client, collection: str):
        self.settings = settings
        self.client = client
        self.collection = collection
        self.url = f"{settings.qdrant_url}/collections/{collection}"

    def exists(self) -> bool:
        response = self.client.get(self.url)
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return True

    def ensure_collection(self) -> None:
        if not self.exists():
            response = self.client.put(
                self.url,
                json={
                    "vectors": {"size": self.settings.embedding_dimensions, "distance": "Cosine"},
                },
            )
            response.raise_for_status()

    def post(self, endpoint: str, payload: dict) -> dict:
        response = self.client.post(self.url + endpoint, json=payload)
        response.raise_for_status()
        return response.json()["result"]

    def count(self, filename: str, revision: str) -> int:
        return self.post(
            "/points/count",
            {
                "filter": document_filter(filename, revision),
                "exact": True,
            },
        )["count"]

    def upsert(self, points: list[dict]) -> None:
        response = self.client.put(self.url + "/points?wait=true", json={"points": points})
        response.raise_for_status()

    def delete(self, filename: str, *, keep_revision: str | None = None) -> None:
        selection = document_filter(filename)
        if keep_revision:
            selection["must_not"] = [{"key": "revision", "match": {"value": keep_revision}}]
        self.post("/points/delete?wait=true", {"filter": selection})

    def search(self, vector: list[float], manifest: dict) -> list[dict]:
        # Failed writes may leave unfinished revisions. Only committed revisions are searchable.
        active_revisions = [
            document_filter(filename, entry["revision"]) for filename, entry in manifest.items()
        ]
        if not active_revisions:
            return []
        return self.post(
            "/points/query",
            {
                "query": vector,
                "filter": {"should": active_revisions},
                "limit": self.settings.top_k,
                "score_threshold": self.settings.minimum_score,
                "with_payload": True,
            },
        )["points"]
