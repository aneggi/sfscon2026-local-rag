# Local RAG System

I want to to this speech on SFSCON in Bolzano, and they approved it. My intention is to show how easy is to make a RAG sistem for loacl LLM with a small one which let us to ask very specific quesiton:
Title: Digital Sovereignty with Local RAG
Subtitle: A minimal open-source prototype using open data and open models
Description: Many  companies want to experiment with AI, but they also need to keep  control over their data, infrastructure and model choices. This short  talk presents a minimal Retrieval-Augmented Generation prototype built  with open data, an open-source database and an open model.
   The session walks through the essential RAG pipeline: loading data,  creating chunks, generating embedding, storing them in a searchable  database, retrieving relevant context and producing grounded answers.  The goal is to show how companies can start exploring private AI with a  small, understandable and reproducible open-source setup.

The best option would be a Mistral model but also other open are ok. Consider I have to do it with my laptop without any GPU, so the model should be small.
The talk will be 10-15 minutes...but Now i want we make a short plan to create this RAG local system and make something which can run multiple times while I add some new documents.
Let's keep it simple I need also to learn about it


## Technical requirements
1. Use `ministral-3:3b` as the initial language model. Define the model name in a configuration file.
2. Implement the backend logic in Python.
3. Run the complete system locally using Docker containers.
4. Define the complete environment with Docker Compose, so it can start with one command:
5. 
   ```bash
   docker compose up --build
6. Use persistent local storage for:- downloaded documents;the vector database;downloaded LLM and embedding-model files.

## Pipeline
1. Download documents
   A Python script reads document URLs from a configuration file and downloads them into the docs/ folder. It must support updating or overwriting a document when requested.
2. Ingest documents
   A second Python script processes the docs/ folder. It identifies new and changed documents, extracts text, splits it into chunks, generates embeddings, and stores the chunks in a vector database. 
3. Vector database container
   One Docker container runs the local vector database. It stores document chunks, embeddings, filenames, source URLs, and other metadata.
4. LLM container
   One Docker container runs Ollama with the configured local language model. It also provides an embedding model to create and search document embeddings.
5. Web chat container
   One lightweight web application container provides a chat interface. It retrieves relevant document passages from the vector database, sends them to the local LLM, and displays answers with source references.
   The interface includes:
   - a button to download or update source documents;
   - a button to ingest new or changed documents;
   - a status message for each operation.

## Expected behavior
The model must answer only from retrieved document passages. If the documents do not contain enough information, it must say so clearly. Each answer should show its source document.




