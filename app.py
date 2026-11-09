"""Streamlit is only the interface; pipeline logic lives in local_rag/."""

from dataclasses import asdict

import httpx
import streamlit as st

from local_rag.chat import answer_question
from local_rag.config import load_settings, load_sources
from local_rag.download import download_documents
from local_rag.ingest import ingest_documents

st.set_page_config(page_title="Local RAG · SFSCON", page_icon="📚", layout="wide")
st.title("Digital Sovereignty with Local RAG")
st.caption("Chat with a local model, using your documents as supporting context")

try:
    settings = load_settings()
    configured_sources = load_sources(settings.sources_file)
except (OSError, ValueError, TypeError, KeyError) as error:
    st.error(f"Please check the configuration: {error}")
    st.stop()


def show_operation(messages: list[str]) -> None:
    for message in messages:
        if message.startswith("ERROR"):
            st.error(message)
        else:
            st.success(message)
    if not messages:
        st.info("No documents to process. Add a source URL or a file in docs/ first.")


def render_answer(message: dict) -> None:
    st.write(message["text"])
    st.caption(
        "Includes document sources"
        if message["sources"]
        else "General model answer · no document sources cited"
    )
    for source in message["sources"]:
        page = f" · page {source['page']}" if source.get("page") else ""
        with st.expander(f"Source: {source['filename']}{page} · chunk {source['chunk']}"):
            st.text(source["text"])
            reference = source.get("reference_url") or source.get("source_url")
            if reference:
                st.link_button("Open original source", reference)
            if source.get("attribution"):
                st.caption(f"{source['attribution']} · {source.get('license', '')}")
    if message["retrieved_passages"]:
        with st.expander("Inspect the passages sent to the model"):
            for passage in message["retrieved_passages"]:
                st.caption(
                    f"Passage {passage['passage_id']} · {passage['filename']} · "
                    f"cosine similarity {passage['score']:.3f}"
                )
                st.text(passage["text"])


with st.sidebar:
    st.header("Prepare your documents")
    st.caption(f"Language model: {settings.language_model}")
    st.caption(f"Embedding model: {settings.embedding_model}")
    overwrite = st.checkbox("Overwrite existing downloads", value=False)
    if st.button("1. Download / update documents", width="stretch"):
        try:
            with st.spinner("Downloading configured sources…"):
                show_operation(download_documents(settings, overwrite=overwrite))
        except (httpx.HTTPError, OSError, ValueError, KeyError) as error:
            st.error(f"Download failed: {error}")
    if st.button("2. Ingest new / changed documents", width="stretch"):
        try:
            with st.spinner("Extracting text, creating embeddings, and updating Qdrant…"):
                show_operation(ingest_documents(settings))
        except (httpx.HTTPError, OSError, ValueError, KeyError) as error:
            st.error(f"Ingestion failed: {error}")
    st.caption("After changing documents, ingest again before asking questions.")
    with st.expander("Configured source documents"):
        for source in configured_sources:
            st.text(source.title or source.filename)
            st.caption(source.filename)
    if st.button("Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()

st.info(
    "Ask a general question or a question about your documents. "
    "Use the sidebar to download and index documents. "
    "Each question is independent; include the subject in follow-up questions."
)
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["text"])
        else:
            render_answer(message)

if question := st.chat_input("Ask a question", max_chars=1000):
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving evidence and generating a local answer…"):
                answer = answer_question(settings, question)
            message = {"role": "assistant", **asdict(answer)}
            st.session_state.messages.append(message)
            render_answer(message)
        except (httpx.HTTPError, OSError, ValueError, KeyError) as error:
            st.error(f"Could not answer. Check the service logs and try again. Details: {error}")
