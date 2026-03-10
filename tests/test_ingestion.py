from langchain_core.documents import Document

from backend import ingestion


def test_normalize_metadata_converts_nested_values():
    normalized = ingestion._normalize_metadata(
        {
            "source": "manual",
            "nested": {},
            "flags": [1, True, "x"],
            "skip_me": None,
            "count": 2,
        }
    )

    assert normalized == {
        "source": "manual",
        "nested": "{}",
        "flags": '[1,true,"x"]',
        "count": 2,
    }


def test_index_documents_async_raises_when_batches_fail():
    class FakeVectorStore:
        async def aadd_documents(self, batch):
            raise ValueError("boom")

    documents = [Document(page_content="hello", metadata={"source": "manual"})]

    try:
        import asyncio

        asyncio.run(
            ingestion.index_documents_async(
                documents,
                batch_size=1,
                vectorstore=FakeVectorStore(),
                max_retries=0,
            )
        )
    except RuntimeError as exc:
        assert "Vector store indexing failed" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when all batches fail")
