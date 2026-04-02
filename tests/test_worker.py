from types import SimpleNamespace

from backend import worker


def test_current_attempt_defaults_to_one():
    assert worker._current_attempt(SimpleNamespace(headers=None)) == 1


def test_current_attempt_reads_header():
    assert worker._current_attempt(SimpleNamespace(headers={"x-attempt": 3})) == 3


def test_republish_for_retry_sets_incremented_attempt():
    published = {}

    class FakeChannel:
        def basic_publish(self, *, exchange, routing_key, body, properties):
            published["exchange"] = exchange
            published["routing_key"] = routing_key
            published["body"] = body
            published["properties"] = properties

    worker._republish_for_retry(
        channel=FakeChannel(),
        settings=SimpleNamespace(queue_ingest="ingest"),
        properties=SimpleNamespace(
            headers={"x-attempt": 1},
            content_type="application/json",
            delivery_mode=2,
            message_id="job-123",
            timestamp=123,
            type="ingest_document",
        ),
        body=b'{"job_id":"job-123"}',
        next_attempt=2,
        error="temporary failure",
    )

    assert published["exchange"] == ""
    assert published["routing_key"] == "ingest"
    assert published["body"] == b'{"job_id":"job-123"}'
    assert published["properties"].headers["x-attempt"] == 2
    assert published["properties"].headers["x-max-attempts"] >= 1
    assert published["properties"].headers["x-last-error"] == "temporary failure"
    assert published["properties"].message_id == "job-123"


def test_process_message_routes_terminal_failure_to_failed_queue(monkeypatch):
    updates = []

    class FakeStore:
        def update_job(self, **kwargs):
            updates.append(kwargs)

    class FakeChannel:
        def __init__(self):
            self.acked = []
            self.nacked = []

        def basic_ack(self, *, delivery_tag):
            self.acked.append(delivery_tag)

        def basic_nack(self, *, delivery_tag, requeue):
            self.nacked.append((delivery_tag, requeue))

    body = (
        b'{"job_id":"job-123","submitted_at":"2026-04-02T16:22:05+00:00",'
        b'"kind":"crawl_documentation","version":1,'
        b'"payload":{"url":"https://docs.example.com","max_depth":2,"extract_depth":"basic"}}'
    )

    def fake_handle_delivery(_body):
        raise ValueError("Tavily crawl failed: missing results")

    monkeypatch.setattr(worker, "_handle_delivery", fake_handle_delivery)
    monkeypatch.setattr(worker, "RedisJobStatusStore", lambda: FakeStore())
    monkeypatch.setattr(worker, "_max_job_attempts", lambda: 3)

    channel = FakeChannel()
    worker._process_message(
        channel=channel,
        method=SimpleNamespace(delivery_tag="tag-1"),
        properties=SimpleNamespace(headers={"x-attempt": 3}),
        body=body,
        settings=SimpleNamespace(
            queue_ingest="ingest",
            queue_ingest_failed="ingest.failed",
        ),
    )

    assert updates == [
        {
            "job_id": "job-123",
            "status": "failed",
            "error": "Tavily crawl failed: missing results",
            "queue": "ingest.failed",
        }
    ]
    assert channel.acked == []
    assert channel.nacked == [("tag-1", False)]


def test_process_message_requeues_retry_before_terminal_failure(monkeypatch):
    updates = []
    republished = []

    class FakeStore:
        def update_job(self, **kwargs):
            updates.append(kwargs)

    class FakeChannel:
        def __init__(self):
            self.acked = []
            self.nacked = []

        def basic_ack(self, *, delivery_tag):
            self.acked.append(delivery_tag)

        def basic_nack(self, *, delivery_tag, requeue):
            self.nacked.append((delivery_tag, requeue))

    body = (
        b'{"job_id":"job-123","submitted_at":"2026-04-02T16:22:05+00:00",'
        b'"kind":"ingest_document","version":1,'
        b'"payload":{"doc_id":"doc-123","text":"hello","metadata":{}}}'
    )

    def fake_handle_delivery(_body):
        raise RuntimeError("temporary failure")

    def fake_republish_for_retry(**kwargs):
        republished.append(kwargs)

    monkeypatch.setattr(worker, "_handle_delivery", fake_handle_delivery)
    monkeypatch.setattr(worker, "_republish_for_retry", fake_republish_for_retry)
    monkeypatch.setattr(worker, "RedisJobStatusStore", lambda: FakeStore())
    monkeypatch.setattr(worker, "_max_job_attempts", lambda: 3)

    channel = FakeChannel()
    worker._process_message(
        channel=channel,
        method=SimpleNamespace(delivery_tag="tag-2"),
        properties=SimpleNamespace(headers={"x-attempt": 1}),
        body=body,
        settings=SimpleNamespace(
            queue_ingest="ingest",
            queue_ingest_failed="ingest.failed",
        ),
    )

    assert len(republished) == 1
    assert republished[0]["next_attempt"] == 2
    assert republished[0]["error"] == "temporary failure"
    assert updates == [
        {
            "job_id": "job-123",
            "status": "queued",
            "error": "retrying attempt 2 of 3: temporary failure",
            "queue": "ingest",
        }
    ]
    assert channel.acked == ["tag-2"]
    assert channel.nacked == []
