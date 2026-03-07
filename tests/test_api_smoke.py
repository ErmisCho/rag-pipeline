from pathlib import Path
import sys

from fastapi.testclient import TestClient

# Ensure project root is importable when pytest runs from root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import main as api_main


def test_health_returns_ok():
    client = TestClient(api_main.app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"


def test_ask_empty_index_returns_fallback(monkeypatch):
    def fake_search_docs(query: str, top_k: int = 4):
        return []

    monkeypatch.setattr(api_main, "search_docs", fake_search_docs)

    client = TestClient(api_main.app)
    response = client.post("/ask", json={"query": "What is this?"})

    assert response.status_code == 200
    data = response.json()
    assert data.get("answer") == "Not enough evidence in the provided documents."
    assert data.get("citations") == []


def test_ingest_enqueues_job(monkeypatch):
    class FakeMessage:
        job_id = "job-123"

    class FakeSettings:
        queue_ingest = "ingest"

    def fake_load_rabbitmq_settings():
        return FakeSettings()

    def fake_publish_ingest_job(*, doc_id: str, text: str, metadata=None, settings=None):
        assert doc_id == "doc-123"
        assert text == "hello world"
        assert metadata == {"source": "manual"}
        assert settings is not None
        return FakeMessage()

    monkeypatch.setattr(api_main, "load_rabbitmq_settings", fake_load_rabbitmq_settings)
    monkeypatch.setattr(api_main, "publish_ingest_job", fake_publish_ingest_job)

    client = TestClient(api_main.app)
    response = client.post(
        "/ingest",
        json={
            "doc_id": "doc-123",
            "text": "hello world",
            "metadata": {"source": "manual"},
        },
    )

    assert response.status_code == 202
    data = response.json()
    assert data == {
        "status": "queued",
        "job_id": "job-123",
        "queue": "ingest",
    }
