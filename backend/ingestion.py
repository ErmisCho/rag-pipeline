import asyncio
import os
import re
from html import unescape
from pathlib import Path
import ssl
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
import certifi
from dotenv import load_dotenv
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from .logger import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent  # go one folder up
load_dotenv(BASE_DIR / ".env")

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


def _clean_text(text: str) -> str:
    # Strip scripts/styles and HTML tags to reduce noise and length.
    text = re.sub(
        r"<script[^>]*>.*?</script>",
        " ",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(
        r"<style[^>]*>.*?</style>",
        " ",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    boilerplate_phrases = [
        "Powered by",
        "Assistant Responses are generated using AI",
        "Join our",
        "All rights reserved",
    ]
    for phrase in boilerplate_phrases:
        text = text.replace(phrase, "")
    return text


def _is_low_value(text: str) -> bool:
    if len(text) < 200:
        return True
    alpha_chars = sum(1 for ch in text if ch.isalpha())
    return alpha_chars < 100


def _chunk_config() -> Tuple[int, int]:
    chunk_size = int(os.environ.get("CHUNK_SIZE", "1200"))
    chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "150"))
    return chunk_size, chunk_overlap


def _batch_config() -> int:
    return int(os.environ.get("BATCH_SIZE", "50"))


def get_embeddings():
    provider = os.environ.get("EMBEDDINGS_PROVIDER", "gemini").lower()
    if provider == "ollama":
        return OllamaEmbeddings(
            model=os.environ.get("OLLAMA_EMBED_MODEL",
                                 "nomic-embed-text:latest"),
            base_url=os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    return GoogleGenerativeAIEmbeddings(
        google_api_key=os.environ["GEMINI_API_KEY"],
        model=os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001"),
        chunk_size=100,
        retry_min_seconds=10,
    )


def get_vectorstore() -> PineconeVectorStore:
    return PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=get_embeddings()
    )


def chunk_text(text: str, allow_short: bool = False) -> List[Document]:
    chunk_size, chunk_overlap = _chunk_config()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    cleaned = _clean_text(text)
    docs = text_splitter.split_documents([Document(page_content=cleaned, metadata={})])
    if allow_short:
        return [doc for doc in docs if doc.page_content.strip()]
    return [doc for doc in docs if not _is_low_value(doc.page_content)]


async def ingest_text(
    doc_id: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    batch_size: Optional[int] = None,
) -> int:
    logger.info("INGEST TEXT")
    base_metadata = metadata or {}
    source = base_metadata.get("source", doc_id)
    split_docs = chunk_text(text, allow_short=True)
    for i, doc in enumerate(split_docs):
        doc.metadata = {
            **base_metadata,
            "source": source,
            "doc_id": doc_id,
            "chunk_id": i,
        }
    vectorstore = get_vectorstore()
    await index_documents_async(
        split_docs,
        batch_size=batch_size or _batch_config(),
        vectorstore=vectorstore,
    )
    return len(split_docs)


async def index_documents_async(
    documents: List[Document],
    batch_size: int = 50,
    vectorstore: Optional[PineconeVectorStore] = None,
    max_concurrency: int = 1,
    max_retries: int = 2,
):
    """Process documents in batches asynchronously."""
    logger.info("VECTOR STORAGE PHASE")
    logger.info(
        f"VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
    )
    vectorstore = vectorstore or get_vectorstore()

    # Create batches
    batches = [
        documents[i: i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    logger.info(
        f"VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    semaphore = asyncio.Semaphore(max_concurrency)

    async def add_batch(batch: List[Document], batch_num: int):
        async with semaphore:
            for attempt in range(1, max_retries + 2):
                try:
                    await vectorstore.aadd_documents(batch)
                    logger.info(
                        f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
                    )
                    return True
                except Exception as e:
                    if attempt <= max_retries:
                        logger.info(
                            f"VectorStore Indexing: Failed to add batch {batch_num} (attempt {attempt}) - {e}. Retrying..."
                        )
                        await asyncio.sleep(1.5 * attempt)
                        continue
                    logger.info(
                        f"VectorStore Indexing: Failed to add batch {batch_num} - {e}"
                    )
                    return False

    # Process batches with limited concurrency
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        logger.info(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        logger.info(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main async function to orchestrate the entire process."""
    await crawl_and_ingest(
        url="https://python.langchain.com/",
        max_depth=5,
        extract_depth="advanced",
    )


async def crawl_and_ingest(
    url: str,
    max_depth: int = 5,
    extract_depth: str = "advanced",
    batch_size: Optional[int] = None,
) -> Dict[str, int]:
    logger.info("DOCUMENTATION INGESTION PIPELINE")

    logger.info(
        f"TavilyCrawl: Starting to crawl {url} with max_depth={max_depth}, extract_depth={extract_depth}",
    )
    res = tavily_crawl.invoke(
        {
            "url": url,
            "max_depth": max_depth,
            "extract_depth": extract_depth,
        }
    )
    results = res.get("results")
    if not isinstance(results, list):
        logger.error(
            "TavilyCrawl: Unexpected response format, missing results. keys=%s",
            list(res.keys()) if isinstance(res, dict) else type(res),
        )
        if isinstance(res, dict) and res.get("error"):
            logger.error("TavilyCrawl error: %s", res.get("error"))
        raise ValueError("Tavily crawl failed: missing results")

    base_host = urlparse(url).netloc.lower()
    base_domain = ".".join(base_host.split(".")[-2:]) if base_host else ""
    docs_host = f"docs.{base_domain}" if base_domain else ""
    all_docs = []
    skipped = 0
    for tavily_crawl_result_item in results:
        item_url = tavily_crawl_result_item.get("url") or ""
        item_host = urlparse(item_url).netloc.lower()
        if base_host and item_host:
            allowed = (
                item_host == base_host
                or item_host.endswith(f".{base_host}")
                or (docs_host and item_host == docs_host)
            )
        else:
            allowed = True
        if not allowed:
            logger.info("TavilyCrawl: Skipping off-domain url=%s", item_url)
            skipped += 1
            continue
        raw_content = tavily_crawl_result_item.get("raw_content")
        if not isinstance(raw_content, str) or not raw_content.strip():
            logger.warning(
                f"TavilyCrawl: Skipping {tavily_crawl_result_item.get('url')} "
                f"because raw_content is empty or None"
            )
            skipped += 1
            continue

        logger.info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )
        cleaned = _clean_text(raw_content)
        if _is_low_value(cleaned):
            skipped += 1
            continue
        all_docs.append(
            Document(
                page_content=cleaned,
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )
    logger.info(
        "TavilyCrawl: Filtered results base_host=%s base_domain=%s kept=%s skipped=%s",
        base_host or "-",
        base_domain or "-",
        len(all_docs),
        skipped,
    )

    # Split documents into chunks
    chunk_size, chunk_overlap = _chunk_config()
    logger.info("DOCUMENT CHUNKING PHASE")
    logger.info(
        f"Text Splitter: Processing {len(all_docs)} documents with {chunk_size} chunk size and {chunk_overlap} overlap",
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splitted_docs = text_splitter.split_documents(all_docs)
    logger.info(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    await index_documents_async(
        splitted_docs,
        batch_size=batch_size or _batch_config(),
    )

    logger.info("PIPELINE COMPLETE")
    logger.info("Documentation ingestion pipeline finished successfully!")
    logger.info("Summary:")
    logger.info(f"Documents extracted: {len(all_docs)}")
    logger.info(f"Chunks created: {len(splitted_docs)}")
    return {
        "documents": len(all_docs),
        "chunks": len(splitted_docs),
    }


if __name__ == '__main__':

    asyncio.run(main())
