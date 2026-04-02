from pathlib import Path
import sys

from fastapi.testclient import TestClient

from backend.jobs import JobStatusRecord

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


def test_get_job_returns_status(monkeypatch):
    class FakeStore:
        def get_job(self, job_id: str):
            assert job_id == "job-123"
            return JobStatusRecord(
                job_id="job-123",
                kind="ingest_document",
                status="completed",
                queue="ingest",
                created_at="2026-03-07T12:00:00+00:00",
                updated_at="2026-03-07T12:01:00+00:00",
                error=None,
            )

    monkeypatch.setattr(api_main, "RedisJobStatusStore", FakeStore)

    client = TestClient(api_main.app)
    response = client.get("/jobs/job-123")

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job-123",
        "kind": "ingest_document",
        "status": "completed",
        "queue": "ingest",
        "created_at": "2026-03-07T12:00:00+00:00",
        "updated_at": "2026-03-07T12:01:00+00:00",
        "error": None,
    }


def test_get_job_returns_404_when_missing(monkeypatch):
    class FakeStore:
        def get_job(self, job_id: str):
            assert job_id == "missing-job"
            return None

    monkeypatch.setattr(api_main, "RedisJobStatusStore", FakeStore)

    client = TestClient(api_main.app)
    response = client.get("/jobs/missing-job")

    assert response.status_code == 404
    assert response.json() == {"detail": "Job not found"}


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
    class FakeSettings:
        queue_ingest = "ingest"

    def fake_load_rabbitmq_settings():
        return FakeSettings()

    class FakeStore:
        def create_job(self, *, job_id: str, kind: str, queue: str):
            assert job_id == "job-123"
            assert kind == "ingest_document"
            assert queue == "ingest"
            return JobStatusRecord(
                job_id="job-123",
                kind="ingest_document",
                status="queued",
                queue="ingest",
                created_at="2026-03-10T10:00:00+00:00",
                updated_at="2026-03-10T10:00:00+00:00",
                error=None,
            )

        def update_job(self, *, job_id: str, status: str, error=None):
            return JobStatusRecord(
                job_id=job_id,
                kind="ingest_document",
                status=status,
                queue="ingest",
                created_at="2026-03-10T10:00:00+00:00",
                updated_at="2026-03-10T10:00:01+00:00",
                error=error,
            )

    def fake_publish_ingest_job(
        *,
        doc_id: str,
        text: str,
        metadata=None,
        settings=None,
        job_id=None,
    ):
        assert doc_id == "doc-123"
        assert text == "hello world"
        assert metadata == {"source": "manual"}
        assert settings is not None
        assert job_id == "job-123"
    monkeypatch.setattr(api_main, "load_rabbitmq_settings", fake_load_rabbitmq_settings)
    monkeypatch.setattr(api_main, "build_ingest_job_message", lambda **kwargs: type(
        "FakeBuiltMessage",
        (),
        {"job_id": "job-123", "kind": "ingest_document"},
    )())
    monkeypatch.setattr(api_main, "RedisJobStatusStore", FakeStore)
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
        "job_id": "job-123",
        "kind": "ingest_document",
        "status": "queued",
        "queue": "ingest",
        "created_at": "2026-03-10T10:00:00+00:00",
        "updated_at": "2026-03-10T10:00:00+00:00",
        "error": None,
    }


def test_crawl_enqueues_job(monkeypatch):
    class FakeSettings:
        queue_ingest = "ingest"

    def fake_load_rabbitmq_settings():
        return FakeSettings()

    class FakeStore:
        def create_job(self, *, job_id: str, kind: str, queue: str):
            assert job_id == "job-456"
            assert kind == "crawl_documentation"
            assert queue == "ingest"
            return JobStatusRecord(
                job_id="job-456",
                kind="crawl_documentation",
                status="queued",
                queue="ingest",
                created_at="2026-03-10T11:00:00+00:00",
                updated_at="2026-03-10T11:00:00+00:00",
                error=None,
            )

        def update_job(self, *, job_id: str, status: str, error=None):
            return JobStatusRecord(
                job_id=job_id,
                kind="crawl_documentation",
                status=status,
                queue="ingest",
                created_at="2026-03-10T11:00:00+00:00",
                updated_at="2026-03-10T11:00:01+00:00",
                error=error,
            )

    def fake_publish_crawl_job(
        *,
        url: str,
        max_depth: int,
        extract_depth: str,
        settings=None,
        job_id=None,
    ):
        assert url == "https://docs.example.com"
        assert max_depth == 3
        assert extract_depth == "advanced"
        assert settings is not None
        assert job_id == "job-456"
    monkeypatch.setattr(api_main, "load_rabbitmq_settings", fake_load_rabbitmq_settings)
    monkeypatch.setattr(api_main, "build_crawl_job_message", lambda **kwargs: type(
        "FakeBuiltMessage",
        (),
        {"job_id": "job-456", "kind": "crawl_documentation"},
    )())
    monkeypatch.setattr(api_main, "RedisJobStatusStore", FakeStore)
    monkeypatch.setattr(api_main, "publish_crawl_job", fake_publish_crawl_job)

    client = TestClient(api_main.app)
    response = client.post(
        "/crawl",
        json={
            "url": "https://docs.example.com",
            "max_depth": 3,
            "extract_depth": "advanced",
        },
    )

    assert response.status_code == 202
    data = response.json()
    assert data == {
        "job_id": "job-456",
        "kind": "crawl_documentation",
        "status": "queued",
        "queue": "ingest",
        "created_at": "2026-03-10T11:00:00+00:00",
        "updated_at": "2026-03-10T11:00:00+00:00",
        "error": None,
    }
