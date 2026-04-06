import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict, Optional

import pika

from .logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class RabbitMQSettings:
    host: str
    port: int
    user: str
    password: str
    vhost: str
    queue_ingest: str
    exchange_dead_letter: str
    queue_ingest_failed: str


@dataclass(frozen=True)
class IngestJobPayload:
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CrawlJobPayload:
    url: str
    max_depth: int
    extract_depth: str


@dataclass(frozen=True)
class IngestJobMessage:
    job_id: str
    submitted_at: str
    payload: IngestJobPayload
    kind: str = "ingest_document"
    version: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class CrawlJobMessage:
    job_id: str
    submitted_at: str
    payload: CrawlJobPayload
    kind: str = "crawl_documentation"
    version: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)


def load_rabbitmq_settings() -> RabbitMQSettings:
    queue_ingest = os.environ["RABBITMQ_QUEUE_INGEST"]
    return RabbitMQSettings(
        host=os.environ["RABBITMQ_HOST"],
        port=int(os.environ["RABBITMQ_PORT"]),
        user=os.environ["RABBITMQ_USER"],
        password=os.environ["RABBITMQ_PASSWORD"],
        vhost=os.environ["RABBITMQ_VHOST"],
        queue_ingest=queue_ingest,
        exchange_dead_letter=os.environ.get(
            "RABBITMQ_EXCHANGE_DEAD_LETTER",
            f"{queue_ingest}.dead_letter",
        ),
        queue_ingest_failed=os.environ.get(
            "RABBITMQ_QUEUE_INGEST_FAILED",
            f"{queue_ingest}.failed",
        ),
    )


def build_ingest_job_message(
    *,
    doc_id: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    job_id: Optional[str] = None,
) -> IngestJobMessage:
    return IngestJobMessage(
        job_id=job_id or str(uuid.uuid4()),
        submitted_at=datetime.now(UTC).isoformat(),
        payload=IngestJobPayload(
            doc_id=doc_id,
            text=text,
            metadata=metadata or {},
        ),
    )


def build_connection_parameters(
    settings: Optional[RabbitMQSettings] = None,
) -> pika.ConnectionParameters:
    resolved = settings or load_rabbitmq_settings()
    credentials = pika.PlainCredentials(
        username=resolved.user,
        password=resolved.password,
    )
    return pika.ConnectionParameters(
        host=resolved.host,
        port=resolved.port,
        virtual_host=resolved.vhost,
        credentials=credentials,
    )


def build_crawl_job_message(
    *,
    url: str,
    max_depth: int,
    extract_depth: str,
    job_id: Optional[str] = None,
) -> CrawlJobMessage:
    return CrawlJobMessage(
        job_id=job_id or str(uuid.uuid4()),
        submitted_at=datetime.now(UTC).isoformat(),
        payload=CrawlJobPayload(
            url=url,
            max_depth=max_depth,
            extract_depth=extract_depth,
        ),
    )


def declare_job_topology(channel: pika.channel.Channel, settings: RabbitMQSettings) -> None:
    channel.exchange_declare(
        exchange=settings.exchange_dead_letter,
        exchange_type="direct",
        durable=True,
    )
    channel.queue_declare(queue=settings.queue_ingest_failed, durable=True)
    channel.queue_bind(
        exchange=settings.exchange_dead_letter,
        queue=settings.queue_ingest_failed,
        routing_key=settings.queue_ingest_failed,
    )
    channel.queue_declare(
        queue=settings.queue_ingest,
        durable=True,
        arguments={
            "x-dead-letter-exchange": settings.exchange_dead_letter,
            "x-dead-letter-routing-key": settings.queue_ingest_failed,
        },
    )


def parse_job_message(body: bytes | str) -> IngestJobMessage | CrawlJobMessage:
    raw_message = body.decode("utf-8") if isinstance(body, bytes) else body
    data = json.loads(raw_message)
    payload = data["payload"]
    kind = data.get("kind", "ingest_document")

    if kind == "crawl_documentation":
        return CrawlJobMessage(
            job_id=data["job_id"],
            submitted_at=data["submitted_at"],
            kind=kind,
            version=int(data.get("version", 1)),
            payload=CrawlJobPayload(
                url=payload["url"],
                max_depth=int(payload["max_depth"]),
                extract_depth=payload["extract_depth"],
            ),
        )

    return IngestJobMessage(
        job_id=data["job_id"],
        submitted_at=data["submitted_at"],
        kind=kind,
        version=int(data.get("version", 1)),
        payload=IngestJobPayload(
            doc_id=payload["doc_id"],
            text=payload["text"],
            metadata=payload.get("metadata") or {},
        ),
    )


def publish_ingest_job(
    *,
    doc_id: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    settings: Optional[RabbitMQSettings] = None,
    job_id: Optional[str] = None,
) -> IngestJobMessage:
    resolved = settings or load_rabbitmq_settings()
    message = build_ingest_job_message(
        doc_id=doc_id,
        text=text,
        metadata=metadata,
        job_id=job_id,
    )
    max_attempts = int(os.environ.get("WORKER_JOB_MAX_ATTEMPTS", "3"))

    logger.info(
        "queue publish_start queue=%s kind=%s doc_id=%s job_id=%s",
        resolved.queue_ingest,
        message.kind,
        message.payload.doc_id,
        message.job_id,
    )

    connection = pika.BlockingConnection(build_connection_parameters(resolved))
    try:
        channel = connection.channel()
        declare_job_topology(channel, resolved)
        channel.basic_publish(
            exchange="",
            routing_key=resolved.queue_ingest,
            body=message.to_json(),
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=2,
                message_id=message.job_id,
                timestamp=int(datetime.now(UTC).timestamp()),
                type=message.kind,
                headers={
                    "x-attempt": 1,
                    "x-max-attempts": max_attempts,
                },
            ),
        )
    finally:
        connection.close()

    logger.info(
        "queue publish_complete queue=%s kind=%s attempt=%s max_attempts=%s job_id=%s",
        resolved.queue_ingest,
        message.kind,
        1,
        max_attempts,
        message.job_id,
    )

    return message


def publish_crawl_job(
    *,
    url: str,
    max_depth: int,
    extract_depth: str,
    settings: Optional[RabbitMQSettings] = None,
    job_id: Optional[str] = None,
) -> CrawlJobMessage:
    resolved = settings or load_rabbitmq_settings()
    message = build_crawl_job_message(
        url=url,
        max_depth=max_depth,
        extract_depth=extract_depth,
        job_id=job_id,
    )
    max_attempts = int(os.environ.get("WORKER_JOB_MAX_ATTEMPTS", "3"))

    logger.info(
        "queue publish_start queue=%s kind=%s url=%s job_id=%s",
        resolved.queue_ingest,
        message.kind,
        message.payload.url,
        message.job_id,
    )

    connection = pika.BlockingConnection(build_connection_parameters(resolved))
    try:
        channel = connection.channel()
        declare_job_topology(channel, resolved)
        channel.basic_publish(
            exchange="",
            routing_key=resolved.queue_ingest,
            body=message.to_json(),
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=2,
                message_id=message.job_id,
                timestamp=int(datetime.now(UTC).timestamp()),
                type=message.kind,
                headers={
                    "x-attempt": 1,
                    "x-max-attempts": max_attempts,
                },
            ),
        )
    finally:
        connection.close()

    logger.info(
        "queue publish_complete queue=%s kind=%s attempt=%s max_attempts=%s job_id=%s",
        resolved.queue_ingest,
        message.kind,
        1,
        max_attempts,
        message.job_id,
    )

    return message
