"""Load human-readable configuration and reject settings that cannot work."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class DocumentSource:
    filename: str
    url: str
    title: str = ""
    attribution: str = ""
    license: str = ""
    reference_url: str = ""


@dataclass(frozen=True)
class Settings:
    language_model: str
    embedding_model: str
    embedding_dimensions: int
    document_prefix: str
    query_prefix: str
    ollama_url: str
    qdrant_url: str
    timeout_seconds: float
    documents_dir: Path
    state_dir: Path
    sources_file: Path
    chunk_size: int
    chunk_overlap: int
    embedding_batch_size: int
    top_k: int
    minimum_score: float
    context_tokens: int
    maximum_answer_tokens: int
    system_prompt_file: Path
    maximum_download_bytes: int


def load_settings(path: Path | None = None) -> Settings:
    path = path or Path(os.getenv("RAG_CONFIG", PROJECT_ROOT / "config/settings.toml"))
    with path.open("rb") as file:
        config = tomllib.load(file)
    models, services, storage = (config[name] for name in ("models", "services", "storage"))
    chunking, retrieval, generation = (
        config[name] for name in ("chunking", "retrieval", "generation")
    )
    settings = Settings(
        language_model=models["language"],
        embedding_model=models["embedding"],
        embedding_dimensions=models["embedding_dimensions"],
        document_prefix=models["document_prefix"],
        query_prefix=models["query_prefix"],
        ollama_url=os.getenv("OLLAMA_URL", services["ollama_url"]).rstrip("/"),
        qdrant_url=os.getenv("QDRANT_URL", services["qdrant_url"]).rstrip("/"),
        timeout_seconds=services["timeout_seconds"],
        documents_dir=PROJECT_ROOT / storage["documents"],
        state_dir=PROJECT_ROOT / storage["state"],
        sources_file=Path(os.getenv("RAG_SOURCES", PROJECT_ROOT / "config/sources.toml")),
        chunk_size=chunking["size"],
        chunk_overlap=chunking["overlap"],
        embedding_batch_size=chunking["embedding_batch_size"],
        top_k=retrieval["top_k"],
        minimum_score=retrieval["minimum_score"],
        context_tokens=generation["context_tokens"],
        maximum_answer_tokens=generation["maximum_answer_tokens"],
        system_prompt_file=PROJECT_ROOT / generation["system_prompt"],
        maximum_download_bytes=int(config["downloads"]["maximum_megabytes"] * 1024 * 1024),
    )
    if not 0 <= settings.chunk_overlap < settings.chunk_size:
        raise ValueError("Chunk overlap must be nonnegative and smaller than chunk size.")
    for name in (
        "embedding_dimensions",
        "timeout_seconds",
        "embedding_batch_size",
        "top_k",
        "context_tokens",
        "maximum_answer_tokens",
        "maximum_download_bytes",
    ):
        if getattr(settings, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if not -1 <= settings.minimum_score <= 1:
        raise ValueError("minimum_score must be between -1 and 1 for cosine similarity.")
    if settings.maximum_answer_tokens >= settings.context_tokens:
        raise ValueError("Answer token limit must be smaller than the context window.")
    return settings


def load_sources(path: Path) -> list[DocumentSource]:
    with path.open("rb") as file:
        sources = [DocumentSource(**source) for source in tomllib.load(file).get("documents", [])]
    filenames = set()
    for source in sources:
        name = source.filename
        if (
            not name
            or name.startswith(".")
            or "/" in name
            or "\\" in name
            or Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS
        ):
            raise ValueError(f"Use a plain .txt, .md, or .pdf filename: {name!r}")
        if name in filenames:
            raise ValueError(f"Duplicate source filename: {name}")
        filenames.add(name)
        for url in (source.url, source.reference_url):
            if url and (urlparse(url).scheme not in {"http", "https"} or not urlparse(url).netloc):
                raise ValueError(f"Source URLs must use http or https: {url}")
    return sources
