"""Download configured sources; --overwrite explicitly refreshes existing files."""

import argparse
import os
import tempfile

import httpx

from local_rag.config import Settings, load_settings, load_sources
from local_rag.state import pipeline_lock


def download_documents(settings: Settings, *, overwrite: bool = False) -> list[str]:
    messages = []
    sources = load_sources(settings.sources_file)
    with pipeline_lock(settings.state_dir):
        settings.documents_dir.mkdir(parents=True, exist_ok=True)
        with httpx.Client(follow_redirects=True, timeout=60, trust_env=False) as client:
            for source in sources:
                destination = settings.documents_dir / source.filename
                if destination.exists() and not overwrite:
                    messages.append(f"Skipped {source.filename}: already downloaded.")
                    continue
                temporary = None
                try:
                    with client.stream("GET", source.url) as response:
                        response.raise_for_status()
                        content_type = response.headers.get("content-type", "").lower()
                        if "text/html" in content_type:
                            raise ValueError("Received HTML; use a direct document URL.")
                        with tempfile.NamedTemporaryFile(
                            dir=settings.documents_dir, suffix=".part", delete=False
                        ) as file:
                            temporary = file.name
                            size = 0
                            for block in response.iter_bytes():
                                size += len(block)
                                if size > settings.maximum_download_bytes:
                                    raise ValueError("Document exceeds the download size limit.")
                                file.write(block)
                        if size == 0:
                            raise ValueError("Server returned an empty document.")
                        if destination.suffix.lower() == ".pdf":
                            with open(temporary, "rb") as file:
                                if not file.read(5).startswith(b"%PDF-"):
                                    raise ValueError("The URL did not return a PDF.")
                        os.replace(temporary, destination)
                    messages.append(f"Downloaded {source.filename} ({size:,} bytes).")
                except (httpx.HTTPError, OSError, ValueError) as error:
                    messages.append(f"ERROR {source.filename}: {error}")
                finally:
                    if temporary and os.path.exists(temporary):
                        os.unlink(temporary)
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Refresh existing documents.")
    arguments = parser.parse_args()
    messages = download_documents(load_settings(), overwrite=arguments.overwrite)
    print("\n".join(messages) or "No document URLs configured.")
    if any(message.startswith("ERROR") for message in messages):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
