"""
ViForge Dataset Ingestion Adapters (Local JSONL/Parquet, HuggingFace, Git).
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List
import pandas as pd
from viforge.utils.logging import logger


class DatasetAdapter:
    """Base interface and file ingestion adapter with streaming and batching support."""

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
    def stream_local_records(
        cls, file_path: Path, chunk_size: int = 1000
    ) -> Iterator[List[Dict[str, Any]]]:
        """Stream local records in chunks to prevent high memory usage."""
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".jsonl":
            chunk: List[Dict[str, Any]] = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        chunk.append(json.loads(stripped))
                        if len(chunk) >= chunk_size:
                            yield chunk
                            chunk = []
            if chunk:
                yield chunk
        elif suffix in [".parquet", ".pq"]:
            try:
                import pyarrow.parquet as pq

                parquet_file = pq.ParquetFile(str(file_path))
                for batch in parquet_file.iter_batches(batch_size=chunk_size):
                    yield batch.to_pandas().to_dict(orient="records")
            except Exception:
                df = pd.read_parquet(file_path)
                for i in range(0, len(df), chunk_size):
                    yield df.iloc[i : i + chunk_size].to_dict(orient="records")
        elif suffix == ".json":
            records = cls.load_local_records(file_path)
            for i in range(0, len(records), chunk_size):
                yield records[i : i + chunk_size]
        else:
            raise ValueError(f"Unsupported dataset format: {suffix}")

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

    @classmethod
    def append_records_jsonl(cls, records: List[Dict[str, Any]], output_path: Path) -> None:
        """Append a batch of records directly to a JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
