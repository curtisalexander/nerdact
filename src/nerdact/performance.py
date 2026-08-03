"""Reproducible warm-inference cost measurements for model comparisons."""

from __future__ import annotations

import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _synchronize(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and the other supported Unix platforms report KiB.
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _accelerator_memory(device: str) -> tuple[int, str]:
    import torch

    if device.startswith("cuda"):
        return torch.cuda.max_memory_reserved(), "CUDA reserved"
    if device.startswith("mps"):
        return torch.mps.driver_allocated_memory(), "MPS driver allocated"
    return 0, "process RSS"


def _cached_snapshot_bytes(model: str, revision: str | None) -> int:
    """Return the unique bytes downloaded in the cache snapshot used by Transformers."""
    from transformers.utils import CONFIG_NAME, cached_file

    config_path = cached_file(model, CONFIG_NAME, revision=revision, local_files_only=True)
    if config_path is None:
        return 0
    snapshot = Path(config_path).parent
    seen: set[tuple[int, int]] = set()
    total = 0
    for path in snapshot.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity not in seen:
            seen.add(identity)
            total += stat.st_size
    return total


def _processor() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def environment_metadata(device: str) -> dict[str, str]:
    import torch
    import transformers

    return {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": _processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device,
    }


def measure_warm_inference(adapter: Any, texts: list[str], repeats: int = 3) -> dict[str, Any]:
    """Warm once, then time sequential single-example predictions over a fixed corpus."""
    if not texts or repeats < 1:
        raise ValueError("profiling requires text and at least one repeat")

    adapter.predict(texts[0])  # Load the model and warm framework/model initialization.
    model = adapter._load().model
    device = str(next(model.parameters()).device)
    _synchronize(device)
    peak_accelerator_bytes, memory_kind = _accelerator_memory(device)

    durations: list[float] = []
    predictions = []
    for repeat in range(repeats):
        current = []
        for text in texts:
            _synchronize(device)
            started = time.perf_counter()
            current.append(adapter.predict(text))
            _synchronize(device)
            accelerator_bytes, memory_kind = _accelerator_memory(device)
            peak_accelerator_bytes = max(peak_accelerator_bytes, accelerator_bytes)
            durations.append(time.perf_counter() - started)
        if repeat == 0:
            predictions = current

    total_seconds = sum(durations)
    peak_rss_bytes = _peak_rss_bytes()
    return {
        "predictions": predictions,
        "performance": {
            "warm_latency_median_ms": statistics.median(durations) * 1000,
            "examples_per_second": len(durations) / total_seconds,
            "characters_per_second": repeats * sum(map(len, texts)) / total_seconds,
            "cached_snapshot_bytes": _cached_snapshot_bytes(adapter.model_name, adapter.revision),
            "peak_memory_bytes": peak_accelerator_bytes or peak_rss_bytes,
            "peak_memory_kind": memory_kind,
            "peak_rss_bytes": peak_rss_bytes,
            "warmup_examples": 1,
            "timed_repeats": repeats,
            "timed_examples": len(durations),
        },
        "environment": environment_metadata(device),
    }
