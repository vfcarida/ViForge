"""
ViForge Dataset Ingestion Adapters (Local JSONL/Parquet, HuggingFace, Git).
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from viforge.utils.logging import logger


class DatasetAdapter:
    """Base interface and file ingestion adapter."""

    @classmethod
    def load_local_records(cls, file_path: Path) -> List[Dict[str, Any]]:
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {file_path}")

        records: List[Dict[str, Any]] = []
        suffix = file_path.suffix.lower()

        if suffix == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        elif suffix in [".parquet", ".pq"]:
            df = pd.read_parquet(file_path)
            records = df.to_dict(orient="records")
        elif suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                records = data if isinstance(data, list) else [data]
        else:
            raise ValueError(f"Unsupported dataset format: {suffix}")

        logger.info(f"Loaded {len(records)} records from {file_path}")
        return records

    @classmethod
    def save_records(cls, records: List[Dict[str, Any]], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()

        if suffix in [".parquet", ".pq"]:
            df = pd.DataFrame(records)
            df.to_parquet(output_path, index=False)
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

        logger.info(f"Saved {len(records)} records to {output_path}")
        return output_path
