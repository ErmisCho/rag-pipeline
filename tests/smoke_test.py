"""
Smoke Test for the RAG Pipeline
--------------------------------

1. Runs the ingestion pipeline (Tavily -> chunking -> embeddings -> Pinecone).
2. Runs a retrieval + generation call against the existing index.

Execute with:
    uv run pytest
"""

from backend.ingestion import main as ingestion_main  # async main()
from backend.core import run_llm
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is importable when pytest runs from root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_smoke_pipeline():
    load_dotenv()

    # 1. Run ingestion (async)
    asyncio.run(ingestion_main())

    # 2. Run retrieval + generation
    question = "What does this pipeline do?"
    response = run_llm(question)

    # Basic contract checks
    assert isinstance(response, dict)
    assert "result" in response
    assert isinstance(response["result"], str)
    assert len(response["result"].strip()) > 0

    # Optional: verify the echo of the query
    assert "query" in response
    assert response["query"] == question
