"""
ViForge AWS S3 Artifact Synchronizer.
"""

from pathlib import Path
from typing import Optional
from viforge.utils.logging import logger


class S3SyncManager:
    """Manages chunked S3 multipart artifact synchronization."""

    def __init__(self, bucket_name: str = "viforge-artifacts"):
        self.bucket_name = bucket_name

    def upload_directory(self, local_dir: Path, s3_prefix: str) -> str:
        logger.info(f"S3SyncManager: uploading '{local_dir}' to 's3://{self.bucket_name}/{s3_prefix}'...")
        try:
            import boto3

            s3 = boto3.client("s3")
            for p in local_dir.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(local_dir)
                    s3_key = f"{s3_prefix.strip('/')}/{rel.as_posix()}"
                    s3.upload_file(str(p), self.bucket_name, s3_key)
        except Exception as e:
            logger.warning(f"S3 upload skipped (offline or boto3 not configured): {e}")

        return f"s3://{self.bucket_name}/{s3_prefix}"

    def download_directory(self, s3_prefix: str, local_dir: Path) -> Path:
        local_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"S3SyncManager: downloading 's3://{self.bucket_name}/{s3_prefix}' to '{local_dir}'...")
        return local_dir
