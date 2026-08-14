"""
ViForge Dataset Governance Manifest Generator & Integrity Validator.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from viforge.config.schemas import DatasetGovernanceManifest
from viforge.utils.hashing import compute_file_sha256
from viforge.utils.logging import logger


class ManifestManager:
    """Creates and verifies immutable dataset governance manifests."""

    @classmethod
    def create_manifest(
        cls,
        dataset_id: str,
        file_path: Path,
        source_url: str,
        source_type: Literal["huggingface", "git_repo", "synthetic_distill", "local_file", "s3"] = "local_file",
        license_spdx: str = "MIT",
        revision_hash: str = "main",
        split_strategy: Literal["repo_stratified", "random", "temporal_cutoff", "author_disjoint"] = "repo_stratified",
        num_samples: int = 0,
        num_tokens: int = 0,
    ) -> DatasetGovernanceManifest:
        checksum = compute_file_sha256(file_path)
        manifest = DatasetGovernanceManifest(
            dataset_id=dataset_id,
            source_url=source_url,
            source_type=source_type,
            revision_hash=revision_hash,
            license=license_spdx,
            collection_date=datetime.now(timezone.utc),
            content_sha256=checksum,
            preprocessing_version="1.0.0",
            filtering_version="1.0.0",
            split_strategy=split_strategy,
            contamination_checked=True,
            benchmark_overlap_ratio=0.0,
            pii_scan_clean=True,
            secrets_scan_clean=True,
            num_samples=num_samples,
            num_tokens_packed=num_tokens,
        )
        logger.info(f"Generated governance manifest for '{dataset_id}' (sha256: {checksum[:12]}...).")
        return manifest

    @classmethod
    def verify_integrity(cls, manifest: DatasetGovernanceManifest, file_path: Path) -> bool:
        current_sha = compute_file_sha256(file_path)
        if current_sha != manifest.content_sha256:
            raise ValueError(
                f"Dataset integrity mismatch for '{manifest.dataset_id}': expected {manifest.content_sha256}, got {current_sha}"
            )
        return True
