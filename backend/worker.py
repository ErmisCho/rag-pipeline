import asyncio
import os
import time
from typing import Any

import pika

from .ingestion import crawl_and_ingest, ingest_text
from .jobs import RedisJobStatusStore
from .logger import get_logger
from .rabbitmq import (
    build_connection_parameters,
    load_rabbitmq_settings,
    parse_job_message,
)

logger = get_logger(__name__)


def _connect_with_retry(settings):
    attempts = int(os.environ.get("WORKER_RABBITMQ_CONNECT_RETRIES", "20"))
    delay_seconds = float(
        os.environ.get("WORKER_RABBITMQ_RETRY_DELAY_SECONDS", "2")
    )

    for attempt in range(1, attempts + 1):
        try:
            logger.info(
                "worker connecting attempt=%s host=%s port=%s",
                attempt,
                settings.host,
                settings.port,
            )
            return pika.BlockingConnection(build_connection_parameters(settings))
        except pika.exceptions.AMQPConnectionError:
            if attempt == attempts:
                raise
            logger.warning(
                "worker connection retrying in %ss attempt=%s",
                delay_seconds,
                attempt,
            )
            time.sleep(delay_seconds)

    raise RuntimeError("worker failed to connect to RabbitMQ")


def _max_job_attempts() -> int:
    return max(1, int(os.environ.get("WORKER_JOB_MAX_ATTEMPTS", "3")))


def _current_attempt(properties: Any) -> int:
    headers = getattr(properties, "headers", None) or {}
    return int(headers.get("x-attempt", 1))


def _republish_for_retry(
    *,
    channel: Any,
    settings: Any,
    properties: Any,
    body: bytes,
    next_attempt: int,
) -> None:
    headers = dict(getattr(properties, "headers", None) or {})
    headers["x-attempt"] = next_attempt
    channel.basic_publish(
        exchange="",
        routing_key=settings.queue_ingest,
        body=body,
        properties=pika.BasicProperties(
            content_type=getattr(properties, "content_type", "application/json"),
            delivery_mode=getattr(properties, "delivery_mode", 2),
            message_id=getattr(properties, "message_id", None),
            timestamp=getattr(properties, "timestamp", None),
            type=getattr(properties, "type", None),
            headers=headers,
        ),
    )


def _handle_delivery(body: bytes) -> None:
    message = parse_job_message(body)
    store = RedisJobStatusStore()
    store.update_job(job_id=message.job_id, status="running")

    if message.kind == "ingest_document":
        logger.info(
            "worker received job_id=%s kind=%s doc_id=%s",
            message.job_id,
            message.kind,
            message.payload.doc_id,
        )
        asyncio.run(
            ingest_text(
                doc_id=message.payload.doc_id,
                text=message.payload.text,
                metadata=message.payload.metadata,
            )
        )
        logger.info(
            "worker completed job_id=%s doc_id=%s",
            message.job_id,
            message.payload.doc_id,
        )
        store.update_job(job_id=message.job_id, status="completed")
        return

    if message.kind == "crawl_documentation":
        logger.info(
            "worker received job_id=%s kind=%s url=%s",
            message.job_id,
            message.kind,
            message.payload.url,
        )
        asyncio.run(
            crawl_and_ingest(
                url=message.payload.url,
                max_depth=message.payload.max_depth,
                extract_depth=message.payload.extract_depth,
            )
        )
        logger.info(
            "worker completed job_id=%s url=%s",
            message.job_id,
            message.payload.url,
        )
        store.update_job(job_id=message.job_id, status="completed")
        return

    raise ValueError(f"Unsupported job kind: {message.kind}")


def run_worker() -> None:
    settings = load_rabbitmq_settings()
    connection = _connect_with_retry(settings)
    channel = connection.channel()
    channel.queue_declare(queue=settings.queue_ingest, durable=True)
    channel.basic_qos(prefetch_count=1)

    def on_message(channel: Any, method: Any, properties: Any, body: bytes) -> None:
        try:
            _handle_delivery(body)
        except Exception as exc:
            try:
                message = parse_job_message(body)
                attempt = _current_attempt(properties)
                max_attempts = _max_job_attempts()
                if attempt < max_attempts:
                    next_attempt = attempt + 1
                    _republish_for_retry(
                        channel=channel,
                        settings=settings,
                        properties=properties,
                        body=body,
                        next_attempt=next_attempt,
                    )
                    RedisJobStatusStore().update_job(
                        job_id=message.job_id,
                        status="queued",
                        error=f"retrying attempt {next_attempt} of {max_attempts}: {exc}",
                    )
                    logger.warning(
                        "worker retrying job_id=%s next_attempt=%s max_attempts=%s",
                        message.job_id,
                        next_attempt,
                        max_attempts,
                    )
                    channel.basic_ack(delivery_tag=method.delivery_tag)
                    return

                RedisJobStatusStore().update_job(
                    job_id=message.job_id,
                    status="failed",
                    error=str(exc),
                )
            except Exception:
                logger.exception("worker failed to persist job status")
            logger.exception("worker failed queue=%s", settings.queue_ingest)
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return

        channel.basic_ack(delivery_tag=method.delivery_tag)

    logger.info(
        "worker listening queue=%s host=%s port=%s",
        settings.queue_ingest,
        settings.host,
        settings.port,
    )
    channel.basic_consume(queue=settings.queue_ingest, on_message_callback=on_message)
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logger.info("worker stopping")
    finally:
        if channel.is_open:
            channel.close()
        connection.close()


if __name__ == "__main__":
    run_worker()
