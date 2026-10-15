"""Transactional, redacted sample checkpoints with an exclusive process lock.

Completed attempts are immutable. Interrupted attempts are never replayed
without explicit opt-in: a target may already have performed a side effect.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from agentic_evalkit.errors import DatasetIntegrityError, ManifestValidationError
from agentic_evalkit.models import EvalRunResult, SampleResult
from agentic_evalkit.reporters.base import DEFAULT_REDACTION_POLICY, apply_redaction

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType
    from typing import BinaryIO

    from agentic_evalkit.models import EvalRunManifest, EvalSample, ResolvedDataset


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


class ProcessLock:
    """An OS lock released even when the owner process crashes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file: BinaryIO | None = None

    def __enter__(self) -> ProcessLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if sys.platform == "win32":
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            raise ManifestValidationError(
                message="run directory is locked by another process"
            ) from error
        self._file = handle
        return self

    def __exit__(self, *args: object) -> None:
        if self._file is not None:
            # Closing the descriptor releases either platform's advisory lock.
            self._file.close()
            self._file = None


class RunCheckpoint:
    def __init__(
        self, directory: Path, *, resume: bool = False, retry_interrupted: bool = False
    ) -> None:
        self.directory = directory
        self.resume = resume
        self.retry_interrupted = retry_interrupted
        self.run_id = str(uuid4())
        self.started_at = datetime.now(UTC)
        self._lock = ProcessLock(directory / ".checkpoint.lock")
        self._connection: sqlite3.Connection | None = None
        self._manifest: EvalRunManifest | None = None
        self._dataset: ResolvedDataset | None = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("checkpoint must be opened as a context manager")
        return self._connection

    def __enter__(self) -> RunCheckpoint:
        self._lock.__enter__()
        try:
            path = self.directory / "checkpoint.sqlite3"
            exists = path.exists()
            if exists and not self.resume:
                raise ManifestValidationError(
                    message="checkpoint exists; use resume or a new run directory"
                )
            if not exists and self.resume:
                raise ManifestValidationError(message="cannot resume: checkpoint does not exist")
            self._connection = sqlite3.connect(path)
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.executescript(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
                "CREATE TABLE IF NOT EXISTS attempts ("
                "sample_id TEXT NOT NULL, attempt INTEGER NOT NULL, status TEXT NOT NULL,"
                "payload TEXT, checksum TEXT, executions INTEGER NOT NULL DEFAULT 0,"
                "PRIMARY KEY (sample_id, attempt));"
            )
            saved = dict(self.connection.execute("SELECT key, value FROM metadata"))
            if exists:
                self.run_id = saved["run_id"]
                self.started_at = datetime.fromisoformat(saved["started_at"])
            else:
                with self.connection:
                    self.connection.executemany(
                        "INSERT INTO metadata VALUES (?, ?)",
                        (
                            ("run_id", self.run_id),
                            ("started_at", self.started_at.isoformat()),
                        ),
                    )
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._lock.__exit__()

    def begin(
        self, manifest: EvalRunManifest, dataset: ResolvedDataset, samples: tuple[EvalSample, ...]
    ) -> None:
        identity = digest(
            {
                "manifest": manifest.model_dump(mode="json"),
                "dataset": dataset.model_dump(mode="json"),
                "samples": [sample.model_dump(mode="json") for sample in samples],
            }
        )
        saved = self.connection.execute(
            "SELECT value FROM metadata WHERE key='identity'"
        ).fetchone()
        if saved is not None and saved[0] != identity:
            raise DatasetIntegrityError(
                message="checkpoint inputs or provenance changed; start a new run"
            )
        if len({sample.sample_id for sample in samples}) != len(samples):
            raise ManifestValidationError(message="duplicate sample ids cannot be checkpointed")
        inflight = self.connection.execute(
            "SELECT COUNT(*) FROM attempts WHERE status='running'"
        ).fetchone()[0]
        if inflight and not self.retry_interrupted:
            raise ManifestValidationError(
                message=(
                    f"{inflight} interrupted attempt(s) may have performed side effects; "
                    "resume with --retry-interrupted only after checking target isolation"
                )
            )
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO metadata VALUES ('identity', ?)", (identity,)
            )
            self.connection.execute("UPDATE attempts SET status='pending' WHERE status='running'")
        self._manifest, self._dataset = manifest, dataset

    def completed(self) -> dict[tuple[str, int], SampleResult]:
        results: dict[tuple[str, int], SampleResult] = {}
        rows = self.connection.execute(
            "SELECT sample_id, attempt, payload, checksum FROM attempts WHERE status='completed'"
        )
        for sample_id, attempt, payload, checksum in rows:
            if digest(payload) != checksum:
                raise DatasetIntegrityError(message="checkpoint result checksum mismatch")
            result = SampleResult.model_validate_json(payload)
            if result.sample.sample_id != sample_id or result.execution.attempt != attempt:
                raise DatasetIntegrityError(message="checkpoint result identity mismatch")
            results[sample_id, attempt] = result
        return results

    def start(self, sample_id: str, attempt: int) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO attempts(sample_id, attempt, status, executions) "
                "VALUES (?, ?, 'running', 1) "
                "ON CONFLICT(sample_id, attempt) DO UPDATE "
                "SET status='running', executions=executions+1 "
                "WHERE status != 'completed'",
                (sample_id, attempt),
            )

    def save(self, result: SampleResult) -> None:
        if self._manifest is None or self._dataset is None:
            raise RuntimeError("begin must precede save")
        envelope = EvalRunResult(
            run_id=self.run_id,
            manifest=self._manifest,
            resolved_dataset=self._dataset,
            samples=(result,),
            started_at=self.started_at,
        )
        redacted = apply_redaction(envelope, DEFAULT_REDACTION_POLICY).samples[0]
        payload = redacted.model_dump_json()
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE attempts SET status='completed', payload=?, checksum=? "
                "WHERE sample_id=? AND attempt=? AND status='running'",
                (payload, digest(payload), result.sample.sample_id, result.execution.attempt),
            )
            if cursor.rowcount != 1:
                raise DatasetIntegrityError(
                    message="checkpoint attempt was not running or is already complete"
                )
