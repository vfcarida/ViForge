"""
ViForge Hashing and Integrity Verification Utilities.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def compute_file_sha256(file_path: Path) -> str:
    """Calculate deterministic SHA-256 hash of a file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_dict_hash(data: Dict[str, Any]) -> str:
    """Calculate deterministic SHA-256 hash of a dictionary."""
    serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
