# RAG Pipeline – End-to-End Retrieval-Augmented Generation System

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline.
It crawls external documentation, ingests and embeds content using Google Gemini, stores vector embeddings in Pinecone, and exposes a Streamlit interface for natural‑language question answering with source citations.

The goal is to demonstrate a modular, maintainable, and realistic RAG system similar to production environments.

![Demo](assets/demo.gif)

---

## Features

### Web Crawling and Ingestion
- Uses Tavily for crawling external URLs
- Extracts content and applies recursive chunking
- Generates embeddings with Google’s `text-embedding-004` model (configurable)
- Stores embeddings in Pinecone for vector search

### Retrieval and LLM Orchestration
- Retrieves the most relevant chunks from Pinecone
- Generates structured answers using Gemini
- Includes source citations for transparency
- Clean, modular retrieval pipeline in `backend/core.py`

### Streamlit Interface
- Minimal UI for interacting with the RAG system
- Chat history preservation
- Clear presentation of answers and sources

### Engineering Practices
- Separation of ingestion, retrieval, and UI components
- Environment-driven configuration via `.env`
- Logging utilities for debugging and monitoring
- Optional LangSmith tracing support

---

## Architecture

```text
Crawling --> Chunking --> Embeddings --> Pinecone Vector Store
                                             |
                                             v
                                      Retriever + LLM
                                             |
                                             v
                                       Streamlit UI
```

---

## Project Structure

```text
rag-pipeline/
  backend/
    core.py          # RAG pipeline logic (retrieval + generation)
    ingestion.py     # Crawling, chunking, embeddings, indexing
    logger.py        # Logging helpers
    __init__.py
  app/
    main.py          # Streamlit UI
    __init__.py
  .env.example
  README.md
  uv.lock
```

---

## Environment Configuration

Create your local environment file:

```bash
cp .env.example .env
```

Fill in your values:

```bash
# LangSmith (optional)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=rag-pipeline
LANGSMITH_API_KEY=<api_key>

# Tavily for crawling
TAVILY_API_KEY=<api_key>

# Google Gemini (LLM and embeddings)
GEMINI_API_KEY=<api_key>
EMBEDDING_MODEL=text-embedding-004

# Pinecone vector database
PINECONE_API_KEY=<api_key>
INDEX_NAME=<index_name>
```

---

## Installation and Running

### Running the Project

Activate your virtual environment and start the application:

### Running the Project (using uv)

# Install dependencies (automatically creates .venv)
```bash
uv sync
```

# Run the ingestion script
```bash
uv run python backend/ingestion.py
```

# Start the Streamlit application
```bash
uv run streamlit run app/main.py
```


---

## How It Works

1. **Ingestion**
   - Crawls target URLs
   - Splits text into overlapping chunks
   - Generates vector embeddings
   - Writes embeddings and metadata to Pinecone

2. **Retrieval**
   - Queries Pinecone for relevant chunks
   - Sends them to the RAG chain

3. **Generation**
   - Gemini synthesizes an answer
   - Adds citations for verification

4. **User Interface**
   - Streamlit provides a chat-like UI

---

## Future Improvements

- Add Dockerfile for containerized deployment
- Support additional vector stores (OpenSearch, pgvector)
- Add retrieval-quality evaluation
- Add automated tests for ingestion and retrieval
- Add background tasks for large-scale ingestion
- Optionally expose a FastAPI backend
