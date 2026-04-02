import base64
import json
import os
import time
import urllib.error
import urllib.parse
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


def _rabbitmq_vhost() -> str:
    return os.environ.get("RABBITMQ_VHOST", "/")


def _failed_queue_name() -> str:
    return os.environ.get("RABBITMQ_QUEUE_INGEST_FAILED", "ingest.failed")


def _max_job_attempts() -> int:
    return int(os.environ.get("WORKER_JOB_MAX_ATTEMPTS", "3"))


def _http_json(*, method: str, url: str, payload: dict | None = None, auth: tuple[str, str] | None = None) -> dict | list:
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
    if os.environ.get("RUN_DLQ_INTEGRATION") != "1":
        pytest.skip(
            "set RUN_DLQ_INTEGRATION=1 to run live DLQ integration tests")

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


def _wait_for_failed_job(job_id: str, timeout_seconds: int = 30) -> dict:
    deadline = time.time() + timeout_seconds
    last_record = None
    while time.time() < deadline:
        last_record = _http_json(
            method="GET",
            url=f"{_api_base_url()}/jobs/{job_id}",
        )
        if last_record["status"] == "failed":
            return last_record
        time.sleep(1)
    raise AssertionError(
        f"job {job_id} did not reach failed state within {timeout_seconds}s; "
        f"last record: {last_record}"
    )


def _get_failed_queue_messages(count: int = 20) -> list[dict]:
    vhost = urllib.parse.quote(_rabbitmq_vhost(), safe="")
    queue = urllib.parse.quote(_failed_queue_name(), safe="")
    url = f"{_rabbitmq_management_base_url()}/queues/{vhost}/{queue}/get"
    payload = {
        "count": count,
        "ackmode": "ack_requeue_true",
        "encoding": "auto",
        "truncate": 50000,
    }
    result = _http_json(
        method="POST",
        url=url,
        payload=payload,
        auth=(_rabbitmq_user(), _rabbitmq_password()),
    )
    assert isinstance(result, list)
    return result


@pytest.mark.integration
def test_failed_job_is_dead_lettered_end_to_end():
    _require_live_stack()

    create_response = _http_json(
        method="POST",
        url=f"{_api_base_url()}/crawl",
        payload={
            "url": "not-a-real-url",
            "max_depth": 2,
            "extract_depth": "basic",
        },
    )
    assert isinstance(create_response, dict)
    job_id = create_response["job_id"]

    final_record = _wait_for_failed_job(job_id)

    assert final_record["job_id"] == job_id
    assert final_record["kind"] == "crawl_documentation"
    assert final_record["status"] == "failed"
    assert final_record["queue"] == _failed_queue_name()
    assert "missing results" in (final_record.get("error") or "")

    messages = _get_failed_queue_messages()
    matching = []
    for message in messages:
        payload = message.get("payload")
        headers = (message.get("properties") or {}).get("headers") or {}
        if isinstance(payload, str) and f'"job_id":"{job_id}"' in payload:
            matching.append((message, headers))

    assert matching, f"job {job_id} was not found in failed queue {_failed_queue_name()}"

    target_message, headers = matching[0]
    assert target_message["routing_key"] == _failed_queue_name()
    assert int(headers.get("x-attempt", 0)) == _max_job_attempts()
    assert int(headers.get("x-max-attempts", 0)) == _max_job_attempts()
    assert "missing results" in str(headers.get("x-last-error", ""))
