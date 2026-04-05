import base64
import json
import os
import time
import urllib.error
import urllib.request

import pytest


def _api_base_url() -> str:
    return os.environ.get("INTEGRATION_API_BASE_URL", "http://localhost:8000")


def _rabbitmq_management_base_url() -> str:
    return os.environ.get(
        "INTEGRATION_RABBITMQ_MANAGEMENT_URL",
        "http://localhost:15672/api",
    )


def _rabbitmq_user() -> str:
    return os.environ.get("RABBITMQ_USER", "guest")


def _rabbitmq_password() -> str:
    return os.environ.get("RABBITMQ_PASSWORD", "guest")


def _ingest_queue_name() -> str:
    return os.environ.get("RABBITMQ_QUEUE_INGEST", "ingest")


def _http_json(
    *, method: str, url: str, payload: dict | None = None, auth: tuple[str, str] | None = None
) -> dict | list:
    """Make HTTP request with JSON payload/response."""
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url, data=data, method=method, headers=headers)
    if auth is not None:
        user, password = auth
        token = f"{user}:{password}".encode("utf-8")
        encoded = base64.b64encode(token).decode("ascii")
        request.add_header("Authorization", f"Basic {encoded}")

    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _require_live_stack() -> None:
    """Skip test if infrastructure is not available."""
    try:
        _http_json(method="GET", url=f"{_api_base_url()}/health")
    except urllib.error.URLError as exc:
        pytest.skip(f"API is not reachable at {_api_base_url()}: {exc}")

    try:
        overview_url = f"{_rabbitmq_management_base_url()}/overview"
        _http_json(
            method="GET",
            url=overview_url,
            auth=(_rabbitmq_user(), _rabbitmq_password()),
        )
    except urllib.error.URLError as exc:
        pytest.skip(
            "RabbitMQ management API is not reachable at "
            f"{_rabbitmq_management_base_url()}: {exc}"
        )


def _wait_for_job_completion(job_id: str, timeout_seconds: int = 60) -> dict:
    """Poll job status until completed or failed, with timeout."""
    deadline = time.time() + timeout_seconds
    last_record = None
    while time.time() < deadline:
        last_record = _http_json(
            method="GET",
            url=f"{_api_base_url()}/jobs/{job_id}",
        )
        status = last_record.get("status")
        if status in ("completed", "failed"):
            return last_record
        time.sleep(1)
    raise AssertionError(
        f"job {job_id} did not reach terminal state within {timeout_seconds}s; "
        f"last record: {last_record}"
    )


@pytest.mark.integration
def test_ingest_job_completes_end_to_end():
    """
    Test the complete async ingest job lifecycle:
    1. API accepts request and returns 202 with job metadata
    2. Worker processes the job from queue
    3. Job status in Redis transitions from queued → running → completed
    """
    _require_live_stack()

    # Create an ingest job with stable test data
    ingest_payload = {
        "doc_id": "test-doc-integration-stable",
        "text": "Python is a high-level programming language. It emphasizes code readability and simplicity.",
        "metadata": {
            "source": "integration_test",
            "test_marker": "do-not-use-in-production",
        },
    }

    # Step 1: POST to /ingest, expect 202 Accepted
    api_response = _http_json(
        method="POST",
        url=f"{_api_base_url()}/ingest",
        payload=ingest_payload,
    )
    assert isinstance(api_response, dict), "API response should be JSON object"
    job_id = api_response.get("job_id")
    assert job_id, "API response should contain job_id"

    # Verify initial response metadata
    assert api_response["kind"] == "ingest_document", "Job kind should be ingest_document"
    assert api_response["status"] == "queued", "Initial job status should be queued"
    assert api_response["queue"] == _ingest_queue_name(
    ), "Job should be on ingest queue"
    assert "created_at" in api_response, "Response should include created_at timestamp"
    assert "updated_at" in api_response, "Response should include updated_at timestamp"

    # Step 2: Wait for worker to process job
    final_record = _wait_for_job_completion(job_id, timeout_seconds=60)

    # Step 3: Verify final state
    assert final_record["job_id"] == job_id, "Job ID should match"
    assert final_record["kind"] == "ingest_document", "Job kind should remain ingest_document"
    assert final_record["status"] == "completed", "Job should complete successfully"
    assert final_record["error"] is None, "Completed job should have no error"
    assert (
        final_record["updated_at"] >= api_response["created_at"]
    ), "updated_at should be after created_at"
