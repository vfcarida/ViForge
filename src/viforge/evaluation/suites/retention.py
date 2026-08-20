"""
ViForge Generalist Capability Retention Benchmark Suites (MMLU-Pro, GSM8K, ARC-Challenge).
Loads real datasets from Hugging Face with deterministic offline fallbacks and verifies responses against gold labels.
"""

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from viforge.config.schemas import BenchmarkResult, SamplingParams
from viforge.inference.backends import BaseInferenceBackend
from viforge.utils.logging import logger


class MMLUProRetentionSuite:
    """MMLU-Pro reasoning benchmark to measure general knowledge retention across 10-choice questions."""

    @property
    def name(self) -> str:
        return "mmlu_pro"

    @staticmethod
    def _extract_answer(completion: str) -> str:
        """Extract choice letter (A-J) from completion."""
        # 1. Look for explicit pattern: "Answer: (B)" or "Choice is B" or "Option B"
        m = re.search(
            r"\b(?:answer|choice|option)\s*(?:is|:)?\s*\(?([A-J])\)?",
            completion,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).upper()

        # 2. Look for parenthesized letter like "(B)"
        m_paren = re.search(r"\(([A-J])\)", completion)
        if m_paren:
            return m_paren.group(1).upper()

        # 3. Look for standalone capital letter choice
        m2 = re.search(r"\b([A-J])\b", completion)
        if m2:
            return m2.group(1).upper()

        cleaned = completion.strip().upper()
        return cleaned[:1] if (cleaned and cleaned[0] in "ABCDEFGHIJ") else ""

    def evaluate_completion(self, completion: str, gold_answer: str) -> bool:
        extracted = self._extract_answer(completion)
        return extracted.strip().upper() == gold_answer.strip().upper()

    def _fallback_toy_problems(self) -> List[Dict[str, Any]]:
        return [
            {
                "question": "What is the primary role of mitochondria in eukaryotic cells?",
                "options": [
                    "Protein synthesis",
                    "ATP generation",
                    "Lipid breakdown",
                    "DNA replication",
                ],
                "answer": "B",
                "prompt": "Question: What is the primary role of mitochondria in eukaryotic cells?\n(A) Protein synthesis\n(B) ATP generation\n(C) Lipid breakdown\n(D) DNA replication\nAnswer:",
            },
            {
                "question": "If a matrix A has determinant 0, which of the following is true?",
                "options": [
                    "A is invertible",
                    "A is non-singular",
                    "A has linearly dependent rows",
                    "Trace(A) = 0",
                ],
                "answer": "C",
                "prompt": "Question: If a matrix A has determinant 0, which of the following is true?\n(A) A is invertible\n(B) A is non-singular\n(C) A has linearly dependent rows\n(D) Trace(A) = 0\nAnswer:",
            },
        ]

    def load_problems(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        problems: List[Dict[str, Any]] = []
        try:
            from datasets import load_dataset

            ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
            for row in ds:
                options = row.get("options", [])
                opt_str = "\n".join(f"({chr(65 + i)}) {opt}" for i, opt in enumerate(options))
                question = row.get("question", "")
                prompt = f"Question: {question}\n{opt_str}\nAnswer:"
                problems.append(
                    {
                        "question": question,
                        "options": options,
                        "answer": str(row.get("answer", "A")).strip().upper(),
                        "prompt": prompt,
                    }
                )
            logger.info(f"Loaded {len(problems)} MMLU-Pro problems from HuggingFace.")
        except Exception as e:
            logger.debug(f"MMLU-Pro dataset loading fallback: {e}")
            problems = self._fallback_toy_problems()

        return problems[:limit] if limit else problems

    def evaluate(
        self,
        inference_backend: BaseInferenceBackend,
        output_dir: Path,
        sampling_params: SamplingParams,
        timeout_seconds: int = 15,
        limit: Optional[int] = None,
    ) -> BenchmarkResult:
        start_time = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)

        test_problems = self.load_problems(limit=limit)
        prompts = [p["prompt"] for p in test_problems]
        completions = inference_backend.generate(prompts, sampling_params)

        passed = 0
        for prob, comp in zip(test_problems, completions):
            if self.evaluate_completion(comp, prob["answer"]):
                passed += 1

        total = len(test_problems)
        accuracy = passed / max(1, total)
        elapsed = time.time() - start_time

        logger.info(f"MMLU-Pro retention evaluation: {passed}/{total} correct ({accuracy:.1%}).")

        return BenchmarkResult(
            benchmark_name=self.name,
            pass_at_k={"acc": round(accuracy, 4)},
            total_problems=total,
            passed_problems=passed,
            failed_problems=total - passed,
            execution_time_seconds=round(elapsed, 2),
            raw_metrics={"accuracy_pct": round(accuracy * 100.0, 2)},
        )


class GSM8KRetentionSuite:
    """GSM8K grade school math benchmark to measure mathematical reasoning retention."""

    @property
    def name(self) -> str:
        return "gsm8k"

    @staticmethod
    def _extract_answer(completion: str) -> str:
        """Extract numeric answer from completion (after '####' or trailing number)."""
        if "####" in completion:
            ans = completion.split("####")[-1].strip()
            return ans.replace(",", "").replace("$", "").strip()

        # Find numbers in text
        cleaned = completion.replace(",", "").replace("$", "")
        matches = re.findall(r"[-+]?\d*\.?\d+", cleaned)
        if matches:
            return matches[-1].strip()

        return completion.strip()

    def evaluate_completion(self, completion: str, gold_answer: str) -> bool:
        extracted = self._extract_answer(completion)
        gold = gold_answer.strip().replace(",", "").replace("$", "")
        try:
            return float(extracted) == float(gold)
        except (ValueError, TypeError):
            return extracted.lower() == gold.lower()

    def _fallback_toy_problems(self) -> List[Dict[str, Any]]:
        return [
            {
                "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether?",
                "answer": "72",
                "prompt": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether?\nAnswer:",
            },
            {
                "question": "Weng earns $12 an hour for babysitting. Yesterday, she babysat for 50 minutes. How much did she earn?",
                "answer": "10",
                "prompt": "Question: Weng earns $12 an hour for babysitting. Yesterday, she babysat for 50 minutes. How much did she earn?\nAnswer:",
            },
        ]

    def load_problems(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        problems: List[Dict[str, Any]] = []
        try:
            from datasets import load_dataset

            ds = load_dataset("openai/gsm8k", "main", split="test")
            for row in ds:
                raw_ans = row.get("answer", "")
                gold_num = raw_ans.split("####")[-1].strip() if "####" in raw_ans else raw_ans
                question = row.get("question", "")
                problems.append(
                    {
                        "question": question,
                        "answer": gold_num,
                        "prompt": f"Question: {question}\nAnswer:",
                    }
                )
            logger.info(f"Loaded {len(problems)} GSM8K problems from HuggingFace.")
        except Exception as e:
            logger.debug(f"GSM8K dataset loading fallback: {e}")
            problems = self._fallback_toy_problems()

        return problems[:limit] if limit else problems

    def evaluate(
        self,
        inference_backend: BaseInferenceBackend,
        output_dir: Path,
        sampling_params: SamplingParams,
        timeout_seconds: int = 15,
        limit: Optional[int] = None,
    ) -> BenchmarkResult:
        start_time = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)

        test_problems = self.load_problems(limit=limit)
        prompts = [p["prompt"] for p in test_problems]
        completions = inference_backend.generate(prompts, sampling_params)

        passed = 0
        for prob, comp in zip(test_problems, completions):
            if self.evaluate_completion(comp, prob["answer"]):
                passed += 1

        total = len(test_problems)
        acc = passed / max(1, total)
        elapsed = time.time() - start_time

        logger.info(f"GSM8K retention evaluation: {passed}/{total} correct ({acc:.1%}).")

        return BenchmarkResult(
            benchmark_name=self.name,
            pass_at_k={"acc": round(acc, 4)},
            total_problems=total,
            passed_problems=passed,
            failed_problems=total - passed,
            execution_time_seconds=round(elapsed, 2),
            raw_metrics={"accuracy_pct": round(acc * 100.0, 2)},
        )


class ARCChallengeRetentionSuite:
    """ARC-Challenge reasoning benchmark."""

    @property
    def name(self) -> str:
        return "arc_challenge"

    @staticmethod
    def _extract_answer(completion: str) -> str:
        """Extract choice letter (A-D) or number (1-4) mapped to letter."""
        # 1. Look for explicit pattern: "Answer: (B)" or "Choice is B" or "Option B"
        m = re.search(
            r"\b(?:answer|choice|option)\s*(?:is|:)?\s*\(?([A-D])\)?",
            completion,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).upper()

        # 2. Look for parenthesized letter like "(B)"
        m_paren = re.search(r"\(([A-D])\)", completion)
        if m_paren:
            return m_paren.group(1).upper()

        # 3. Look for numeric choices 1-4
        m_num = re.search(
            r"\b(?:option|choice|answer)?\s*(?:is|:)?\s*([1-4])\b",
            completion,
            re.IGNORECASE,
        )
        if m_num:
            return chr(65 + int(m_num.group(1)) - 1)

        # 4. Look for standalone capital letter
        m2 = re.search(r"\b([A-D])\b", completion)
        if m2:
            return m2.group(1).upper()

        cleaned = completion.strip().upper()
        return cleaned[:1] if (cleaned and cleaned[0] in "ABCD") else ""

    def evaluate_completion(self, completion: str, gold_answer: str) -> bool:
        extracted = self._extract_answer(completion)
        gold = gold_answer.strip().upper()
        # Handle numeric gold answers mapped to A-D
        if gold in ("1", "2", "3", "4"):
            gold = chr(65 + int(gold) - 1)
        return extracted == gold

    def _fallback_toy_problems(self) -> List[Dict[str, Any]]:
        return [
            {
                "question": "Which property of an object most affects how it responds to the force of gravity?",
                "choices": ["mass", "shape", "volume", "color"],
                "answer": "A",
                "prompt": "Question: Which property of an object most affects how it responds to the force of gravity?\n(A) mass (B) shape (C) volume (D) color\nAnswer:",
            },
        ]

    def load_problems(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        problems: List[Dict[str, Any]] = []
        try:
            from datasets import load_dataset

            ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
            for row in ds:
                choices = row.get("choices", {})
                labels = choices.get("label", [])
                texts = choices.get("text", [])
                opt_str = " ".join(f"({lbl}) {txt}" for lbl, txt in zip(labels, texts))
                question = row.get("question", "")
                problems.append(
                    {
                        "question": question,
                        "choices": texts,
                        "answer": str(row.get("answerKey", "A")).strip().upper(),
                        "prompt": f"Question: {question}\n{opt_str}\nAnswer:",
                    }
                )
            logger.info(f"Loaded {len(problems)} ARC-Challenge problems from HuggingFace.")
        except Exception as e:
            logger.debug(f"ARC-Challenge dataset loading fallback: {e}")
            problems = self._fallback_toy_problems()

        return problems[:limit] if limit else problems

    def evaluate(
        self,
        inference_backend: BaseInferenceBackend,
        output_dir: Path,
        sampling_params: SamplingParams,
        timeout_seconds: int = 15,
        limit: Optional[int] = None,
    ) -> BenchmarkResult:
        start_time = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)

        test_problems = self.load_problems(limit=limit)
        prompts = [p["prompt"] for p in test_problems]
        completions = inference_backend.generate(prompts, sampling_params)

        passed = 0
        for prob, comp in zip(test_problems, completions):
            if self.evaluate_completion(comp, prob["answer"]):
                passed += 1

        total = len(test_problems)
        acc = passed / max(1, total)
        elapsed = time.time() - start_time

        logger.info(f"ARC-Challenge retention evaluation: {passed}/{total} correct ({acc:.1%}).")

        return BenchmarkResult(
            benchmark_name=self.name,
            pass_at_k={"acc": round(acc, 4)},
            total_problems=total,
            passed_problems=passed,
            failed_problems=total - passed,
            execution_time_seconds=round(elapsed, 2),
            raw_metrics={"accuracy_pct": round(acc * 100.0, 2)},
        )
