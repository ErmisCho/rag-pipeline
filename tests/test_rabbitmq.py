import json

from backend.rabbitmq import (
    CrawlJobMessage,
    RabbitMQSettings,
    build_connection_parameters,
    build_crawl_job_message,
    build_ingest_job_message,
    load_rabbitmq_settings,
    parse_job_message,
)


def test_load_rabbitmq_settings_from_env(monkeypatch):
    monkeypatch.setenv("RABBITMQ_HOST", "rabbitmq")
    monkeypatch.setenv("RABBITMQ_PORT", "5672")
    monkeypatch.setenv("RABBITMQ_USER", "guest")
    monkeypatch.setenv("RABBITMQ_PASSWORD", "guest")
    monkeypatch.setenv("RABBITMQ_VHOST", "/")
    monkeypatch.setenv("RABBITMQ_QUEUE_INGEST", "ingest")

    settings = load_rabbitmq_settings()

    assert settings == RabbitMQSettings(
        host="rabbitmq",
        port=5672,
        user="guest",
        password="guest",
        vhost="/",
        queue_ingest="ingest",
    )


def test_build_ingest_job_message_generates_contract():
    message = build_ingest_job_message(
        doc_id="doc-123",
        text="hello world",
        metadata={"source": "manual"},
        job_id="job-123",
    )

    payload = json.loads(message.to_json())

    assert payload["job_id"] == "job-123"
    assert payload["kind"] == "ingest_document"
    assert payload["version"] == 1
    assert payload["payload"] == {
        "doc_id": "doc-123",
        "text": "hello world",
        "metadata": {"source": "manual"},
    }
    assert payload["submitted_at"].endswith("+00:00")


def test_build_connection_parameters_uses_settings():
    settings = RabbitMQSettings(
        host="rabbitmq",
        port=5672,
        user="guest",
        password="guest",
        vhost="/",
        queue_ingest="ingest",
    )

    params = build_connection_parameters(settings)

    assert params.host == "rabbitmq"
    assert params.port == 5672
    assert params.virtual_host == "/"


def test_parse_ingest_job_message_round_trips_payload():
    message = build_ingest_job_message(
        doc_id="doc-123",
        text="hello world",
        metadata={"source": "manual"},
        job_id="job-123",
    )

    parsed = parse_job_message(message.to_json().encode("utf-8"))

    assert parsed.job_id == "job-123"
    assert parsed.kind == "ingest_document"
    assert parsed.payload.doc_id == "doc-123"
    assert parsed.payload.text == "hello world"
    assert parsed.payload.metadata == {"source": "manual"}


def test_parse_crawl_job_message_round_trips_payload():
    message = build_crawl_job_message(
        url="https://docs.example.com",
        max_depth=3,
        extract_depth="advanced",
        job_id="job-456",
    )

    parsed = parse_job_message(message.to_json())

    assert isinstance(parsed, CrawlJobMessage)
    assert parsed.job_id == "job-456"
    assert parsed.kind == "crawl_documentation"
    assert parsed.payload.url == "https://docs.example.com"
    assert parsed.payload.max_depth == 3
    assert parsed.payload.extract_depth == "advanced"
