"""Small filesystem and serialization helpers."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path


def allowed_filename(filename: str, extensions: set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in extensions


def analysis_directory(root: Path, analysis_id: str) -> Path:
    """Return an analysis directory after basic identifier validation."""
    if not analysis_id.isalnum() or len(analysis_id) != 32:
        raise ValueError("Invalid analysis identifier")
    return root / analysis_id


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cleanup_expired_directories(roots: list[Path], retention_hours: int) -> None:
    """Remove only direct per-analysis directories older than the documented retention period."""
    if retention_hours <= 0:
        return
    cutoff = time.time() - retention_hours * 3600
    for root in roots:
        if not root.exists():
            continue
        for candidate in root.iterdir():
            if candidate.is_dir() and len(candidate.name) == 32 and candidate.name.isalnum() and candidate.stat().st_mtime < cutoff:
                shutil.rmtree(candidate, ignore_errors=True)
