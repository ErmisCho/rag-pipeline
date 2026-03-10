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
    )

    assert published["exchange"] == ""
    assert published["routing_key"] == "ingest"
    assert published["body"] == b'{"job_id":"job-123"}'
    assert published["properties"].headers["x-attempt"] == 2
    assert published["properties"].message_id == "job-123"
