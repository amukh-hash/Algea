from __future__ import annotations

import io
import json
import subprocess
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .job_defs import Job


@dataclass
class JobResult:
    status: str
    exit_code: int
    error_summary: str | None
    stdout_path: str
    stderr_path: str
    artifacts: list[str]


class JobRunner:
    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run

    def _run_handler(self, job: Job, context: dict[str, Any], stdout_path: Path, stderr_path: Path) -> JobResult:
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            payload = job.handler(context)
            print(json.dumps(payload, sort_keys=True))
        stdout_path.write_text(out.getvalue(), encoding="utf-8")
        stderr_path.write_text(err.getvalue(), encoding="utf-8")

        if not isinstance(payload, dict):
            raise TypeError(f"job handler '{job.name}' must return dict")
        status = str(payload.get("status", "ok"))
        if status == "failed":
            raise RuntimeError(str(payload.get("summary", "handler returned failed")))

        artifacts_field = payload.get("artifacts", {})
        artifacts: list[str]
        if isinstance(artifacts_field, dict):
            artifacts = [str(v) for v in artifacts_field.values()]
        elif isinstance(artifacts_field, list):
            artifacts = [str(v) for v in artifacts_field]
        else:
            artifacts = []
        return JobResult("success", 0, None, str(stdout_path), str(stderr_path), artifacts)

    def run(self, job: Job, context: dict[str, Any], jobs_dir: Path) -> JobResult:
        jobs_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = jobs_dir / f"{job.name}.stdout.log"
        stderr_path = jobs_dir / f"{job.name}.stderr.log"

        attempts = 0
        delay = 0.5
        while attempts <= job.retries:
            attempts += 1
            started = datetime.now().timestamp()
            try:
                result = self._run_handler(job, context, stdout_path, stderr_path)
                runtime = datetime.now().timestamp() - started
                if runtime > job.timeout_s:
                    raise TimeoutError(f"job exceeded timeout ({runtime:.2f}s > {job.timeout_s}s)")
                return result
            except Exception as exc:
                stderr_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
                if attempts > job.retries:
                    return JobResult("failed", 1, str(exc), str(stdout_path), str(stderr_path), [])
                time.sleep(min(delay, 2.0))
                delay *= 2
        return JobResult("failed", 1, "retry exhausted", str(stdout_path), str(stderr_path), [])

    @staticmethod
    def run_subprocess(command: list[str], stdout_path: Path, stderr_path: Path, timeout_s: int) -> JobResult:
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
            try:
                proc = subprocess.run(command, stdout=out, stderr=err, timeout=timeout_s, check=False, text=True)
            except subprocess.TimeoutExpired:
                return JobResult("failed", 124, "timeout", str(stdout_path), str(stderr_path), [])
        return JobResult(
            "success" if proc.returncode == 0 else "failed",
            proc.returncode,
            None if proc.returncode == 0 else f"exit={proc.returncode}",
            str(stdout_path),
            str(stderr_path),
            [],
        )
