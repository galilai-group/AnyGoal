"""Preserve periodic DriftWorld checkpoints without changing a running trainer."""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch


def inspect_and_snapshot(source: Path, run_directory: Path, multiple: int) -> Path | None:
    """Copy one stable checkpoint inode when its embedded step is periodic."""
    with source.open("rb") as source_handle:
        source_stat = os.fstat(source_handle.fileno())
        checkpoint = torch.load(source_handle, map_location="cpu", weights_only=False)
        step = int(checkpoint["step"])
        del checkpoint
        gc.collect()
        if step % multiple:
            print(f"INSPECTED step={step} source={source}", flush=True)
            return None

        destination = run_directory / f"ckpt-step{step}-periodic20k.pth"
        if destination.exists():
            existing = torch.load(destination, map_location="cpu", weights_only=False)
            existing_step = int(existing["step"])
            del existing
            gc.collect()
            if existing_step != step or destination.stat().st_size != source_stat.st_size:
                raise RuntimeError(f"Existing periodic snapshot is invalid: {destination}")
            print(f"EXISTS step={step} path={destination}", flush=True)
            return destination

        temporary = run_directory / f".{destination.name}.partial-{os.getpid()}"
        source_handle.seek(0)
        try:
            with temporary.open("xb") as destination_handle:
                shutil.copyfileobj(
                    source_handle, destination_handle, length=8 * 1024 * 1024
                )
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    copied = torch.load(destination, map_location="cpu", weights_only=False)
    copied_step = int(copied["step"])
    del copied
    gc.collect()
    if copied_step != step or destination.stat().st_size != source_stat.st_size:
        raise RuntimeError(f"Periodic snapshot verification failed: {destination}")
    print(
        f"SNAPSHOT step={step} bytes={source_stat.st_size} path={destination}",
        flush=True,
    )
    return destination


def jobs_are_active(job_ids: list[str]) -> bool:
    if not job_ids:
        return True
    result = subprocess.run(
        ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", action="append", type=Path, required=True)
    parser.add_argument("--training-job-id", action="append", default=[])
    parser.add_argument("--multiple", type=int, default=20_000)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.multiple < 1 or args.poll_seconds <= 0:
        raise ValueError("multiple and poll-seconds must be positive.")

    run_directories = [path.expanduser().resolve() for path in args.run_directory]
    for run_directory in run_directories:
        if not run_directory.is_dir():
            raise FileNotFoundError(run_directory)

    seen_files: set[tuple[int, int, int, int]] = set()
    while True:
        for run_directory in run_directories:
            for filename in ("ckpt-latest.pth", "ckpt-2nd-latest.pth"):
                source = run_directory / filename
                try:
                    stat = source.stat()
                except FileNotFoundError:
                    continue
                identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                if identity in seen_files:
                    continue
                try:
                    inspect_and_snapshot(source, run_directory, args.multiple)
                except (EOFError, OSError, RuntimeError) as error:
                    print(f"RETRY source={source} error={error!r}", flush=True)
                    continue
                seen_files.add(identity)

        if args.once or not jobs_are_active(args.training_job_id):
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
