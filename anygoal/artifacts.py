"""Versioned sidecars for reproducible datasets, checkpoints, and results."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def _git_commit(directory: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_artifact_sidecar(
    directory: str | Path,
    *,
    artifact_type: str,
    artifact_id: str,
    seed: int,
    command: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Atomically write ``artifact.json`` without changing the primary artifact."""

    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "artifact.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    payload = {
        "format_version": 1,
        "artifact_type": artifact_type,
        "id": artifact_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "command": list(command or sys.argv),
            "seed": int(seed),
            "git_commit": _git_commit(directory),
        },
        "metadata": dict(metadata or {}),
    }
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination
