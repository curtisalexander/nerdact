"""Auditable provenance for generated benchmark artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_provenance(
    root: Path, input_paths: list[Path], options: dict[str, Any]
) -> dict[str, Any]:
    """Fingerprint effective inputs, source, dependencies, options, and Git state."""
    root = root.resolve()

    def relative(path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            return str(resolved)

    inputs = {
        relative(path): _sha256(path)
        for path in sorted({path.resolve() for path in input_paths})
        if path.is_file()
    }
    source_paths = sorted((root / "src" / "nerdact").glob("*.py")) + [
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    source_digest = hashlib.sha256()
    for path in source_paths:
        source_digest.update(relative(path).encode())
        source_digest.update(b"\0")
        source_digest.update(path.read_bytes())
        source_digest.update(b"\0")

    dependencies = {}
    for package in ("nerdact", "torch", "transformers", "datasets", "gliner", "gliner2"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue

    status = _git(root, "status", "--porcelain", "--untracked-files=normal")
    identity = {
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_worktree_dirty": bool(status) if status is not None else None,
        "source_sha256": source_digest.hexdigest(),
        "inputs_sha256": inputs,
        "dependencies": dependencies,
        "options": options,
    }
    run_id = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return {
        "run_id": run_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        **identity,
    }
