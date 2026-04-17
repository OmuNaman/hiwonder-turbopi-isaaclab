from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_parquet_engine() -> str:
    for engine_name in ("pyarrow", "fastparquet"):
        try:
            __import__(engine_name)
            return engine_name
        except Exception:
            continue
    raise RuntimeError(
        "Saving parquet requires `pyarrow` or `fastparquet` in the Isaac Lab Python environment.\n"
        "Install one with:\n"
        "  /workspace/isaaclab/_isaac_sim/python.sh -m pip install pyarrow"
    )


def ensure_trace_dir(output_root: Path | str, trace_name: str) -> Path:
    trace_dir = Path(output_root) / trace_name
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

