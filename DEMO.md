# SFSCON: Digital Sovereignty with Local RAG

**Subtitle:** A minimal open-source prototype using open data and open models.

## Prepare before the session

Start the system, download the sample, and ingest it. Rehearse on your actual
CPU-only laptop. Keep Docker's model and database volumes, and check that chat
still works with networking disconnected. Keep the source document open so you
can compare an answer directly with its evidence.

Prepare a document inside the running container to demonstrate incremental
ingestion. This writes only to the Docker document volume:

```bash
./compose exec -T web sh -c 'cat > /app/docs/workshop-note.txt' <<'EOF'
This is a fictional workshop example, not an official SFSCON policy.
The example library lends development laptops for 14 days.
Borrowers can request one extension of 7 days.
EOF
```

## A 12-minute walkthrough

| Time | Show | Explain |
| --- | --- | --- |
| 0:00–1:30 | The title and the local architecture | Sovereignty means controlling documents, inference, infrastructure, and model choices. |
| 1:30–3:00 | `compose.yaml` and the two TOML files | Three running services, one startup job, and explicit model/source choices. |
| 3:00–4:30 | Download and ingest buttons | Text becomes overlapping chunks; embeddings make passages searchable. |
| 4:30–6:30 | Ask about redistribution including sale | Inspect the answer, source quotation, and retrieved passage. |
| 6:30–7:30 | Ask “Are you an AI?” | The model can answer generally; document-backed answers also show source references. |
| 7:30–9:30 | Create the prepared note in the container and ingest | The original source is skipped; only the new document is embedded. |
| 9:30–10:30 | Ask “For how many days can the example library's laptops be borrowed?” | Retrieval makes a small model useful for a specific corpus. |
| 10:30–12:00 | Change 14 days to 21 days; ingest and ask again | Updating the documents changes the evidence without training the language model. |

For a 10-minute slot, omit the last update and shorten the configuration tour.
For 15 minutes, inspect `documents.py`, `ingest.py`, and the citation validator.
Allow extra time for CPU inference; fill that time by explaining the retrieved
passage or source document.

## Three distinctions to explain clearly

1. **Retrieval is not training.** Documents are indexed; the language model's
   weights stay the same.
2. **Local is a deployment choice.** You still need to choose licenses, data,
   storage, and model versions consciously. Preparation uses network downloads;
   inference uses local services.
3. **A citation is evidence to inspect.** The code resolves selected sources to
   original passages, but cannot mechanically establish that every generated claim is true.

## A useful next learning exercise

Change one setting at a time. Compare smaller chunks with larger chunks, inspect
which passages are retrieved, and record whether the answer and citation behavior
improve. Keep the corpus and questions fixed while comparing settings.
