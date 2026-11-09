"""Index new and changed documents; skip unchanged work and remove deleted files."""

import argparse
import hashlib
import json
from dataclasses import asdict
from uuid import NAMESPACE_URL, uuid5

import httpx
from pypdf.errors import PyPdfError

from local_rag.config import SUPPORTED_EXTENSIONS, Settings, load_settings, load_sources
from local_rag.documents import extract_chunks
from local_rag.ollama import Ollama
from local_rag.state import pipeline_lock, read_json, write_json
from local_rag.vector_store import VectorStore, collection_name


def ingest_documents(settings: Settings, *, force: bool = False) -> list[str]:
    messages = []
    sources = {source.filename: asdict(source) for source in load_sources(settings.sources_file)}
    with (
        pipeline_lock(settings.state_dir),
        httpx.Client(timeout=settings.timeout_seconds, trust_env=False) as client,
    ):
        ollama = Ollama(settings, client)
        collection = collection_name(settings, ollama.embedding_digest())
        store = VectorStore(settings, client, collection)
        store.ensure_collection()
        manifest_path = settings.state_dir / f"{collection}.json"
        manifest = read_json(manifest_path)
        settings.documents_dir.mkdir(parents=True, exist_ok=True)
        paths = sorted(
            path
            for path in settings.documents_dir.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        current_files = {path.relative_to(settings.documents_dir).as_posix() for path in paths}
        for filename in sorted(set(manifest) - current_files):
            # Keep the manifest entry until deletion succeeds, so a failed cleanup can retry.
            store.delete(filename)
            del manifest[filename]
            write_json(manifest_path, manifest)
            messages.append(f"Removed {filename} from the index.")
        for path in paths:
            filename = path.relative_to(settings.documents_dir).as_posix()
            try:
                metadata = sources.get(filename, {"filename": filename})
                content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                revision = hashlib.sha256(
                    (content_hash + json.dumps(metadata, sort_keys=True)).encode()
                ).hexdigest()
                previous = manifest.get(filename, {})
                if (
                    not force
                    and previous.get("revision") == revision
                    and store.count(filename, revision) == previous.get("chunks")
                ):
                    # Retry cleanup if a prior run committed but stopped before deleting old points.
                    store.delete(filename, keep_revision=revision)
                    messages.append(f"Skipped {filename}: unchanged ({previous['chunks']} chunks).")
                    continue
                chunks = extract_chunks(path, settings.chunk_size, settings.chunk_overlap)
                # Generate every embedding before touching the currently usable document.
                vectors = []
                for offset in range(0, len(chunks), settings.embedding_batch_size):
                    batch = chunks[offset : offset + settings.embedding_batch_size]
                    vectors.extend(ollama.embed([chunk.text for chunk in batch]))
                points = [
                    {
                        "id": str(
                            uuid5(NAMESPACE_URL, f"{collection}/{filename}/{revision}/{number}")
                        ),
                        "vector": vector,
                        "payload": {
                            **metadata,
                            "filename": filename,
                            "source_url": metadata.get("url", ""),
                            "content_sha256": content_hash,
                            "revision": revision,
                            "chunk": number,
                            "page": chunk.page,
                            "text": chunk.text,
                            "embedding_model": settings.embedding_model,
                        },
                    }
                    for number, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True), 1)
                ]
                for offset in range(0, len(points), 64):
                    store.upsert(points[offset : offset + 64])
                # The manifest is the commit point. Queries only see complete revisions.
                manifest[filename] = {"revision": revision, "chunks": len(chunks)}
                write_json(manifest_path, manifest)
                store.delete(filename, keep_revision=revision)
                messages.append(f"Indexed {filename}: {len(chunks)} chunks.")
            except (httpx.HTTPError, OSError, ValueError, KeyError, PyPdfError) as error:
                messages.append(f"ERROR {filename}: {error}")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-embed unchanged documents too.")
    arguments = parser.parse_args()
    messages = ingest_documents(load_settings(), force=arguments.force)
    print("\n".join(messages) or "No supported documents found. Download or add documents first.")
    if any(message.startswith("ERROR") for message in messages):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
