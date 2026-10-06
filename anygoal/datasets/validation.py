"""Non-mutating validation for the common AnyGoal HDF5 layout."""

from __future__ import annotations

from pathlib import Path

import hdf5plugin  # noqa: F401  # Register compressed HDF5 filters.
import h5py

from anygoal.types import ValidationIssue, ValidationReport


def validate_hdf5(path: str | Path, *, dataset_id: str | None = None) -> ValidationReport:
    """Inspect an HDF5 dataset without rewriting or normalizing its contents."""

    path = Path(path).expanduser().resolve()
    issues: list[ValidationIssue] = []
    if not path.is_file():
        return ValidationReport(
            path=path,
            dataset_id=dataset_id,
            valid=False,
            issues=(ValidationIssue("error", "File does not exist."),),
        )
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        issues.append(ValidationIssue("error", "Expected an .h5 or .hdf5 file."))

    keys: tuple[str, ...] = ()
    rows: int | None = None
    episodes: int | None = None
    try:
        with h5py.File(path, "r") as handle:
            keys = tuple(sorted(handle.keys()))
            required = {"action", "pixels"}
            missing = sorted(required.difference(handle.keys()))
            if missing:
                issues.append(
                    ValidationIssue(
                        "error", "Missing required top-level keys: " + ", ".join(missing)
                    )
                )
            lengths: dict[str, int] = {}
            for key, value in handle.items():
                if isinstance(value, h5py.Dataset) and value.ndim:
                    lengths[key] = len(value)
            if "action" in lengths:
                rows = lengths["action"]
                for key in ("pixels", "state", "proprio", "episode"):
                    if key in lengths and lengths[key] != rows:
                        issues.append(
                            ValidationIssue(
                                "error",
                                f"Leading dimension mismatch: action={rows}, {key}={lengths[key]}.",
                            )
                        )
            if "ep_len" in handle:
                episodes = len(handle["ep_len"])
                total = int(handle["ep_len"][:].sum())
                if rows is not None and total != rows:
                    issues.append(
                        ValidationIssue(
                            "error", f"ep_len sums to {total}, but action has {rows} rows."
                        )
                    )
            elif "episode" in handle and len(handle["episode"]):
                episodes = len(set(handle["episode"][:].tolist()))
            else:
                issues.append(
                    ValidationIssue(
                        "warning", "No ep_len or episode index was found."
                    )
                )
    except OSError as error:
        issues.append(ValidationIssue("error", f"Could not open HDF5 file: {error}"))

    return ValidationReport(
        path=path,
        dataset_id=dataset_id,
        valid=not any(issue.level == "error" for issue in issues),
        keys=keys,
        rows=rows,
        episodes=episodes,
        issues=tuple(issues),
    )
