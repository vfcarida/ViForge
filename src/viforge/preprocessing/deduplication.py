"""
ViForge MinHash LSH and Exact Deduplication Engine.
"""

import hashlib
import re
from typing import Any, Dict, List, Set, Tuple
import xxhash
from viforge.utils.logging import logger


class MinHashDeduplicator:
    """
    MinHash Locality-Sensitive Hashing (LSH) for scalable near-duplicate code detection.
    """

    def __init__(self, num_perm: int = 128, jaccard_threshold: float = 0.85, ngram_size: int = 3):
        self.num_perm = num_perm
        self.threshold = jaccard_threshold
        self.ngram_size = ngram_size
        self.seeds = [xxhash.xxh64(str(i)).intdigest() for i in range(num_perm)]

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

    def deduplicate_corpus(
        self, records: List[Dict[str, Any]], text_key: str = "text"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        seen_exact_hashes: Set[str] = set()
        seen_signatures: List[List[int]] = []
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
            is_near_dup = False
            for existing_sig in seen_signatures:
                sim = self.estimate_jaccard(sig, existing_sig)
                if sim >= self.threshold:
                    is_near_dup = True
                    near_dups += 1
                    break

            if not is_near_dup:
                seen_exact_hashes.add(exact_hash)
                seen_signatures.append(sig)
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
