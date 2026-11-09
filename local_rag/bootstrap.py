"""Make Compose startup deterministic without relying on curl inside service images."""

import time

import httpx

from local_rag.config import load_settings
from local_rag.ollama import prepare_models


def wait_for_service(client: httpx.Client, url: str, attempts: int = 60) -> None:
    for attempt in range(attempts):
        try:
            client.get(url).raise_for_status()
            return
        except httpx.HTTPError:
            if attempt == attempts - 1:
                raise
            print(f"Waiting for {url}…", flush=True)
            time.sleep(2)


def main() -> None:
    settings = load_settings()
    with httpx.Client(timeout=5, trust_env=False) as client:
        wait_for_service(client, settings.qdrant_url + "/readyz")
        wait_for_service(client, settings.ollama_url + "/api/tags")
    prepare_models(settings)


if __name__ == "__main__":
    main()
