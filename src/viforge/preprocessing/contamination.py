"""
ViForge Contamination and Benchmark Leakage Detector.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from viforge.utils.logging import logger


class ContaminationDetector:
    """
    Scans candidate dataset records against canonical evaluation benchmark n-gram lookup banks.
    """

    def __init__(self, ngram_size: int = 10, max_allowed_overlap: float = 0.05):
        if ngram_size < 1:
            raise ValueError(f"ngram_size must be positive integer >= 1, got {ngram_size}")
        if not (0.0 <= max_allowed_overlap <= 1.0):
            raise ValueError(
                f"max_allowed_overlap must be in range [0.0, 1.0], got {max_allowed_overlap}"
            )
        self.ngram_size = ngram_size
        self.default_max_allowed_overlap = max_allowed_overlap
        self._benchmark_ngrams: Dict[str, Set[str]] = {}

    def _normalize_tokens(self, text: str) -> List[str]:
        text = re.sub(r"#.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
        text = re.sub(r"'''.*?'''", "", text, flags=re.DOTALL)
        return re.findall(r"\b[A-Za-z0-9_]+\b", text.lower())

    def _extract_ngrams(self, tokens: List[str]) -> Set[str]:
        if len(tokens) < self.ngram_size:
            return set(["_".join(tokens)])
        ngrams = set()
        for i in range(len(tokens) - self.ngram_size + 1):
            ngrams.add("_".join(tokens[i : i + self.ngram_size]))
        return ngrams

    def register_benchmark_corpus(self, benchmark_name: str, benchmark_texts: List[str]) -> None:
        bank: Set[str] = set()
        for text in benchmark_texts:
            tokens = self._normalize_tokens(text)
            ngrams = self._extract_ngrams(tokens)
            bank.update(ngrams)
        self._benchmark_ngrams[benchmark_name] = bank
        logger.info(
            f"Registered benchmark '{benchmark_name}' with {len(bank)} unique {self.ngram_size}-grams."
        )

    def check_sample(self, text: str) -> Dict[str, float]:
        tokens = self._normalize_tokens(text)
        sample_ngrams = self._extract_ngrams(tokens)
        if not sample_ngrams:
            return {b: 0.0 for b in self._benchmark_ngrams}

        overlaps = {}
        for b_name, b_bank in self._benchmark_ngrams.items():
            intersection = sample_ngrams.intersection(b_bank)
            overlap_ratio = len(intersection) / len(sample_ngrams)
            overlaps[b_name] = round(overlap_ratio, 4)
        return overlaps

    def filter_dataset(
        self,
        records: List[Dict[str, Any]],
        text_key: str = "text",
        max_allowed_overlap: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        threshold = (
            max_allowed_overlap
            if max_allowed_overlap is not None
            else self.default_max_allowed_overlap
        )
        clean_records = []
        contaminated_count = 0
        leakage_by_benchmark: Dict[str, int] = {b: 0 for b in self._benchmark_ngrams}

        for rec in records:
            text = str(rec.get(text_key, ""))
            overlaps = self.check_sample(text)
            is_contaminated = False

            for b_name, ratio in overlaps.items():
                if ratio > threshold:
                    is_contaminated = True
                    leakage_by_benchmark[b_name] = leakage_by_benchmark.get(b_name, 0) + 1

            if not is_contaminated:
                clean_records.append(rec)
            else:
                contaminated_count += 1

        overall_ratio = contaminated_count / max(1, len(records))
        stats = {
            "total_records": len(records),
            "clean_records": len(clean_records),
            "contaminated_records_removed": contaminated_count,
            "leakage_ratio_pct": round(overall_ratio * 100.0, 2),
            "leakage_by_benchmark": leakage_by_benchmark,
        }
        logger.info(
            f"Contamination scan finished: {len(clean_records)}/{len(records)} clean records retained "
            f"({contaminated_count} dropped)."
        )
        return clean_records, stats
