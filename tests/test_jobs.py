from backend.jobs import JobStatusRecord, RedisJobStatusStore, RedisSettings


class FakeRedis:
    def __init__(self):
        self.data = {}

    def set(self, key: str, value: str):
        self.data[key] = value

    def get(self, key: str):
        return self.data.get(key)


def test_create_job_persists_queued_status():
    store = RedisJobStatusStore(
        client=FakeRedis(),
        settings=RedisSettings(host="redis", port=6379, db=0, key_prefix="job_status"),
    )

    record = store.create_job(job_id="job-123", kind="ingest_document", queue="ingest")

    assert record.job_id == "job-123"
    assert record.kind == "ingest_document"
    assert record.status == "queued"
    assert record.queue == "ingest"
    assert record.error is None


def test_update_job_transitions_status():
    store = RedisJobStatusStore(
        client=FakeRedis(),
        settings=RedisSettings(host="redis", port=6379, db=0, key_prefix="job_status"),
    )
    store.create_job(job_id="job-123", kind="ingest_document", queue="ingest")

    record = store.update_job(job_id="job-123", status="failed", error="boom")

    assert record == JobStatusRecord(
        job_id="job-123",
        kind="ingest_document",
        status="failed",
        queue="ingest",
        created_at=record.created_at,
        updated_at=record.updated_at,
        error="boom",
    )
    assert store.get_job("job-123") == record
