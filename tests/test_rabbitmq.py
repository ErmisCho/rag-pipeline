import json

from backend.rabbitmq import (
    RabbitMQSettings,
    build_connection_parameters,
    build_ingest_job_message,
    load_rabbitmq_settings,
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
