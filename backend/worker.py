import asyncio
from typing import Any

import pika

from .ingestion import ingest_text
from .logger import get_logger
from .rabbitmq import (
    build_connection_parameters,
    load_rabbitmq_settings,
    parse_ingest_job_message,
)

logger = get_logger(__name__)


def _handle_delivery(body: bytes) -> None:
    message = parse_ingest_job_message(body)

    if message.kind != "ingest_document":
        raise ValueError(f"Unsupported job kind: {message.kind}")

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


def run_worker() -> None:
    settings = load_rabbitmq_settings()
    connection = pika.BlockingConnection(build_connection_parameters(settings))
    channel = connection.channel()
    channel.queue_declare(queue=settings.queue_ingest, durable=True)
    channel.basic_qos(prefetch_count=1)

    def on_message(channel: Any, method: Any, properties: Any, body: bytes) -> None:
        try:
            _handle_delivery(body)
        except Exception:
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
