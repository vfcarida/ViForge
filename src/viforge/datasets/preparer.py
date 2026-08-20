"""
ViForge Domain-Agnostic Dataset Preparation Pipeline.
Ingests, filters, deduplicates, and standardizes domain datasets for SFT, DPO, GRPO, and CPT.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from viforge.config.schemas import DatasetConfig, DomainPresetConfig, DomainPresetDatasetItem
from viforge.datasets.adapters import DatasetAdapter
from viforge.datasets.filters import CodeQualityFilter, TextQualityFilter
from viforge.preprocessing.deduplication import MinHashDeduplicator
from viforge.utils.logging import logger


def load_domain_preset(preset_path_or_domain: Union[str, Path]) -> DomainPresetConfig:
    """
    Load a domain preset configuration from YAML file or domain name in `configs/domains/`.
    """
    preset_path = Path(preset_path_or_domain)
    if not preset_path.exists():
        # Look in configs/domains/
        possible = Path("configs") / "domains" / f"{preset_path_or_domain}.yaml"
        if not possible.exists():
            possible = Path("configs") / "domains" / f"{preset_path_or_domain}.yml"
        if possible.exists():
            preset_path = possible
        else:
            raise FileNotFoundError(
                f"Domain preset not found at '{preset_path_or_domain}' or 'configs/domains/{preset_path_or_domain}.yaml'"
            )

    with open(preset_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return DomainPresetConfig(**data)


class DatasetPreparer:
    """
    End-to-end dataset transformation engine that prepares domain data for any training technique.
    """

    def __init__(self, config: DatasetConfig):
        self.config = config

    def _fallback_toy_records(self) -> List[Dict[str, Any]]:
        """Fallback mock records for offline execution or tests."""
        if self.config.format in ("sft", "cpt"):
            return [
                {
                    "instruction": "Write a Python function to calculate fibonacci numbers.",
                    "output": "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)\n",
                    "text": "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)\n",
                },
                {
                    "instruction": "Write a function to check if a number is prime.",
                    "output": "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n",
                    "text": "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n",
                },
                {
                    "instruction": "Write a Python function to reverse a string.",
                    "output": "def reverse_string(s: str) -> str:\n    return s[::-1]\n",
                    "text": "def reverse_string(s: str) -> str:\n    return s[::-1]\n",
                },
            ]
        elif self.config.format == "dpo":
            return [
                {
                    "prompt": "Write a Python function to add two numbers.",
                    "chosen": 'def add(a: int, b: int) -> int:\n    """Add two numbers and return sum."""\n    return a + b\n',
                    "rejected": "def add(a, b):\n    return a - b\n",
                },
                {
                    "prompt": "Write a function to check if list is empty.",
                    "chosen": "def is_empty(lst: list) -> bool:\n    return len(lst) == 0\n",
                    "rejected": "def is_empty(lst):\n    return None\n",
                },
            ]
        elif self.config.format == "grpo":
            return [
                {
                    "prompt": "Write a Python function to multiply two numbers.\ndef multiply(a, b):\n"
                },
                {"prompt": "Write a Python function to square an integer.\ndef square(n):\n"},
            ]
        return [{"text": "def helper():\n    return True\n"}]

    def _load_source(self) -> List[Dict[str, Any]]:
        """Load raw records from local file path or Hugging Face dataset."""
        source_path = Path(self.config.source)
        records: List[Dict[str, Any]] = []

        if source_path.exists():
            records = DatasetAdapter.load_local_records(source_path)
        else:
            try:
                from datasets import load_dataset

                kwargs: Dict[str, Any] = {"split": self.config.split}
                if self.config.subset:
                    ds = load_dataset(self.config.source, self.config.subset, **kwargs)
                else:
                    ds = load_dataset(self.config.source, **kwargs)

                records = [dict(row) for row in ds]
                logger.info(
                    f"Loaded {len(records)} records from HuggingFace '{self.config.source}'."
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load dataset '{self.config.source}': {e}. Using offline fallback."
                )
                records = self._fallback_toy_records()

        # Deterministic shuffle & sample limit
        if records:
            rng = random.Random(self.config.seed)
            rng.shuffle(records)
            if self.config.max_samples and len(records) > self.config.max_samples:
                records = records[: self.config.max_samples]

        return records

    def _apply_quality_filters(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply domain quality heuristics filter."""
        if not self.config.quality_filter:
            return records

        if self.config.domain in ("software_engineering", "code") or "code" in self.config.domain:
            q_filter = CodeQualityFilter(
                min_length=self.config.min_length,
                max_length=self.config.max_length,
                text_column=self.config.text_column,
            )
        else:
            q_filter = TextQualityFilter(
                min_length=self.config.min_length,
                max_length=self.config.max_length,
                text_column=self.config.text_column,
            )

        passed = [r for r in records if q_filter(r)]
        logger.info(
            f"Quality filtering: {len(passed)}/{len(records)} records retained "
            f"({len(records) - len(passed)} filtered out)."
        )
        return passed

    def _deduplicate(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate records using MinHash LSH."""
        if not self.config.dedup or not records:
            return records

        # Extract text representation for deduplication
        temp_records = []
        for r in records:
            rec_copy = dict(r)
            if "text" not in rec_copy:
                # Build unified text representation for deduplication
                if "messages" in rec_copy and isinstance(rec_copy["messages"], list):
                    rec_copy["_dedup_text"] = " ".join(
                        str(m.get("content", ""))
                        for m in rec_copy["messages"]
                        if isinstance(m, dict)
                    )
                elif "prompt" in rec_copy:
                    rec_copy["_dedup_text"] = (
                        str(rec_copy.get("prompt", ""))
                        + " "
                        + str(rec_copy.get("chosen", "") or rec_copy.get("completion", ""))
                    )
                elif "instruction" in rec_copy:
                    rec_copy["_dedup_text"] = (
                        str(rec_copy.get("instruction", ""))
                        + " "
                        + str(rec_copy.get("output", "") or rec_copy.get("response", ""))
                    )
                else:
                    rec_copy["_dedup_text"] = str(r)
            else:
                rec_copy["_dedup_text"] = str(rec_copy["text"])
            temp_records.append(rec_copy)

        deduper = MinHashDeduplicator(jaccard_threshold=self.config.dedup_threshold)
        unique_recs, stats = deduper.deduplicate_corpus(temp_records, text_key="_dedup_text")

        for u in unique_recs:
            u.pop("_dedup_text", None)

        return unique_recs

    def _format_for_technique(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform raw examples into standard schema required for target technique."""
        formatted: List[Dict[str, Any]] = []

        for r in records:
            if self.config.format == "sft":
                # Standard: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
                if "messages" in r and isinstance(r["messages"], list):
                    formatted.append({"messages": r["messages"]})
                else:
                    inst_col = self.config.instruction_column or (
                        "instruction" if "instruction" in r else "prompt"
                    )
                    resp_col = self.config.response_column or (
                        "output"
                        if "output" in r
                        else ("response" if "response" in r else "completion")
                    )

                    user_content = str(r.get(inst_col, ""))
                    if (
                        self.config.input_column
                        and self.config.input_column in r
                        and r[self.config.input_column]
                    ):
                        user_content += f"\n\n{r[self.config.input_column]}"
                    elif "input" in r and r["input"]:
                        user_content += f"\n\n{r['input']}"

                    assistant_content = str(r.get(resp_col, ""))
                    if not user_content and not assistant_content and self.config.text_column in r:
                        user_content = "Complete the following code:"
                        assistant_content = str(r[self.config.text_column])

                    formatted.append(
                        {
                            "messages": [
                                {"role": "user", "content": user_content},
                                {"role": "assistant", "content": assistant_content},
                            ]
                        }
                    )

            elif self.config.format == "dpo":
                # Standard: {"prompt": "...", "chosen": "...", "rejected": "..."}
                p_col = self.config.instruction_column or (
                    "prompt" if "prompt" in r else "instruction"
                )
                c_col = self.config.chosen_column or (
                    "chosen" if "chosen" in r else "chosen_response"
                )
                r_col = self.config.rejected_column or (
                    "rejected" if "rejected" in r else "rejected_response"
                )

                prompt = str(r.get(p_col, ""))
                chosen = str(r.get(c_col, ""))
                rejected = str(r.get(r_col, ""))

                if not chosen and "output" in r:
                    chosen = str(r["output"])
                if not rejected:
                    rejected = "# Incomplete response\n"

                formatted.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

            elif self.config.format == "grpo":
                # Standard: {"prompt": "..."}
                p_col = self.config.instruction_column or (
                    "prompt"
                    if "prompt" in r
                    else ("instruction" if "instruction" in r else self.config.text_column)
                )
                prompt = str(r.get(p_col, ""))
                formatted.append({"prompt": prompt})

            elif self.config.format == "cpt":
                # Standard: {"text": "..."}
                text_val = str(
                    r.get(
                        self.config.text_column,
                        r.get("content", r.get("code", r.get("output", str(r)))),
                    )
                )
                formatted.append({"text": text_val})

        return formatted

    def prepare(self) -> List[Dict[str, Any]]:
        """Execute complete preparation pipeline."""
        raw = self._load_source()
        filtered = self._apply_quality_filters(raw)
        deduped = self._deduplicate(filtered)
        formatted = self._format_for_technique(deduped)
        logger.info(
            f"Dataset preparation complete: {len(formatted)} records formatted for '{self.config.format}'."
        )
        return formatted

    def save(self, output_path: Union[str, Path]) -> Path:
        """Prepare dataset and save to file."""
        records = self.prepare()
        return DatasetAdapter.save_records(records, Path(output_path))
