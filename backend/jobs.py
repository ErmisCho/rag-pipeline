import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Optional

from redis import Redis

from .logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class RedisSettings:
    host: str
    port: int
    db: int
    key_prefix: str


@dataclass(frozen=True)
class JobStatusRecord:
    job_id: str
    kind: str
    status: str
    queue: str
    created_at: str
    updated_at: str
    error: Optional[str] = None


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def load_redis_settings() -> RedisSettings:
    return RedisSettings(
        host=os.environ["REDIS_HOST"],
        port=int(os.environ.get("REDIS_PORT", "6379")),
        db=int(os.environ.get("REDIS_DB", "0")),
        key_prefix=os.environ.get("REDIS_JOB_KEY_PREFIX", "job_status"),
    )


def create_redis_client(settings: Optional[RedisSettings] = None) -> Redis:
    resolved = settings or load_redis_settings()
    return Redis(
        host=resolved.host,
        port=resolved.port,
        db=resolved.db,
        decode_responses=True,
    )


class RedisJobStatusStore:
    def __init__(self, client: Optional[Redis] = None, settings: Optional[RedisSettings] = None):
        self.settings = settings or load_redis_settings()
        self.client = client or create_redis_client(self.settings)

    def _key(self, job_id: str) -> str:
        return f"{self.settings.key_prefix}:{job_id}"

    def create_job(self, *, job_id: str, kind: str, queue: str) -> JobStatusRecord:
        timestamp = utc_now_iso()
        record = JobStatusRecord(
            job_id=job_id,
            kind=kind,
            status="queued",
            queue=queue,
            created_at=timestamp,
            updated_at=timestamp,
        )
        self.client.set(self._key(job_id), json.dumps(asdict(record), separators=(",", ":"), sort_keys=True))
        logger.info(
            "job created job_id=%s kind=%s status=%s queue=%s",
            record.job_id,
            record.kind,
            record.status,
            record.queue,
        )
        return record

    def update_job(
        self,
        *,
        job_id: str,
        status: str,
        error: Optional[str] = None,
        queue: Optional[str] = None,
    ) -> JobStatusRecord:
        current = self.get_job(job_id)
        if current is None:
            raise KeyError(job_id)
        record = JobStatusRecord(
            job_id=current.job_id,
            kind=current.kind,
            status=status,
            queue=queue or current.queue,
            created_at=current.created_at,
            updated_at=utc_now_iso(),
            error=error,
        )
        self.client.set(self._key(job_id), json.dumps(asdict(record), separators=(",", ":"), sort_keys=True))
        logger.info(
            "job updated job_id=%s kind=%s status=%s queue=%s error=%s",
            record.job_id,
            record.kind,
            record.status,
            record.queue,
            record.error,
        )
        return record

    def get_job(self, job_id: str) -> Optional[JobStatusRecord]:
        raw = self.client.get(self._key(job_id))
        if raw is None:
            return None
        return JobStatusRecord(**json.loads(raw))
