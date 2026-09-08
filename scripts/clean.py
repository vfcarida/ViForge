"""
Cross-platform repository cleanup script for ViForge.
Safely removes Python build artifacts, caches, and test run files on Windows, Linux, and macOS.
"""

import shutil
import sys
from pathlib import Path


def clean_repository() -> None:
    root = Path(__file__).resolve().parent.parent
    patterns_dir = [
        "__pycache__",
        "*.egg-info",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "build",
        "dist",
    ]
    patterns_file = [
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".coverage",
        "coverage.xml",
    ]

    removed_dirs = 0
    removed_files = 0

    for pattern in patterns_dir:
        for p in root.rglob(pattern):
            if p.is_dir() and ".venv" not in p.parts and "venv" not in p.parts:
                try:
                    shutil.rmtree(p, ignore_errors=True)
                    removed_dirs += 1
                except Exception:
                    pass

    for pattern in patterns_file:
        for p in root.rglob(pattern):
            if p.is_file() and ".venv" not in p.parts and "venv" not in p.parts:
                try:
                    p.unlink(missing_ok=True)
                    removed_files += 1
                except Exception:
                    pass

    print(f"ViForge clean complete: removed {removed_dirs} directories and {removed_files} files.")


if __name__ == "__main__":
    clean_repository()
