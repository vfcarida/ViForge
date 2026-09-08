"""
ViForge MinHash LSH and Exact Deduplication Engine.
"""

import hashlib
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import xxhash
from viforge.utils.logging import logger


class MinHashDeduplicator:
    """
    MinHash Locality-Sensitive Hashing (LSH) for scalable near-duplicate code detection.
    Uses banded indexing to achieve O(N) expected scaling instead of O(N^2) pairwise comparisons.
    """

    def __init__(
        self,
        num_perm: int = 128,
        jaccard_threshold: float = 0.85,
        ngram_size: int = 3,
        num_bands: Optional[int] = None,
    ):
        self.num_perm = num_perm
        self.threshold = jaccard_threshold
        self.ngram_size = ngram_size
        self.seeds = [xxhash.xxh64(str(i).encode("utf-8")).intdigest() for i in range(num_perm)]

        # Determine bands and rows per band (b * r <= num_perm)
        if num_bands is not None:
            self.num_bands = num_bands
            self.rows_per_band = num_perm // num_bands
        else:
            # Default to 16 bands of 8 rows for 128 permutations
            if num_perm % 16 == 0:
                self.num_bands = 16
                self.rows_per_band = num_perm // 16
            elif num_perm % 8 == 0:
                self.num_bands = 8
                self.rows_per_band = num_perm // 8
            else:
                self.num_bands = 4
                self.rows_per_band = num_perm // 4

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def _get_shingles(self, text: str) -> Set[str]:
        tokens = self._tokenize(text)
        if len(tokens) < self.ngram_size:
            return set([" ".join(tokens)])
        shingles = set()
        for i in range(len(tokens) - self.ngram_size + 1):
            shingles.add(" ".join(tokens[i : i + self.ngram_size]))
        return shingles

    def compute_signature(self, text: str) -> List[int]:
        shingles = self._get_shingles(text)
        if not shingles:
            return [0] * self.num_perm

        signature = [float("inf")] * self.num_perm
        for shingle in shingles:
            shingle_bytes = shingle.encode("utf-8")
            for i, seed in enumerate(self.seeds):
                h = xxhash.xxh64(shingle_bytes, seed=seed).intdigest()
                if h < signature[i]:
                    signature[i] = h

        return [int(x) if x != float("inf") else 0 for x in signature]

    @staticmethod
    def estimate_jaccard(sig_a: List[int], sig_b: List[int]) -> float:
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / len(sig_a)

    def _get_band_hashes(self, signature: List[int]) -> List[int]:
        """Compute 64-bit hash for each band slice in the signature."""
        band_hashes = []
        for b in range(self.num_bands):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            band_slice = signature[start:end]
            h = xxhash.xxh64(str(band_slice).encode("utf-8")).intdigest()
            band_hashes.append(h)
        return band_hashes

    def deduplicate_corpus(
        self, records: List[Dict[str, Any]], text_key: str = "text"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        seen_exact_hashes: Set[str] = set()
        seen_signatures: List[List[int]] = []
        # LSH Hash Table: maps (band_idx, band_hash) -> list of document indices in seen_signatures
        band_buckets: Dict[Tuple[int, int], List[int]] = {}
        unique_records: List[Dict[str, Any]] = []
        exact_dups = 0
        near_dups = 0

        for rec in records:
            text = str(rec.get(text_key, ""))
            exact_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
            if exact_hash in seen_exact_hashes:
                exact_dups += 1
                continue

            sig = self.compute_signature(text)
            band_hashes = self._get_band_hashes(sig)

            # Find candidate duplicates that collide in at least one band bucket
            candidate_ids: Set[int] = set()
            for b_idx, b_hash in enumerate(band_hashes):
                bucket_key = (b_idx, b_hash)
                if bucket_key in band_buckets:
                    candidate_ids.update(band_buckets[bucket_key])

            is_near_dup = False
            for cand_id in candidate_ids:
                sim = self.estimate_jaccard(sig, seen_signatures[cand_id])
                if sim >= self.threshold:
                    is_near_dup = True
                    near_dups += 1
                    break

            if not is_near_dup:
                doc_idx = len(seen_signatures)
                seen_exact_hashes.add(exact_hash)
                seen_signatures.append(sig)
                for b_idx, b_hash in enumerate(band_hashes):
                    bucket_key = (b_idx, b_hash)
                    band_buckets.setdefault(bucket_key, []).append(doc_idx)
                unique_records.append(rec)

        stats = {
            "initial_records": len(records),
            "unique_records": len(unique_records),
            "exact_duplicates_removed": exact_dups,
            "near_duplicates_removed": near_dups,
            "retention_rate_pct": round((len(unique_records) / max(1, len(records))) * 100.0, 2),
        }
        logger.info(
            f"Deduplication complete: {len(unique_records)}/{len(records)} retained "
            f"({exact_dups} exact, {near_dups} near duplicates removed)."
        )
        return unique_records, stats
