"""
Example: Creating and Registering a Custom Domain Benchmark in ViForge.
"""

from pathlib import Path
from viforge.config.schemas import BenchmarkResult, SamplingParams
from viforge.evaluation.suites import evaluator_registry
from viforge.inference.backends import BaseInferenceBackend


class CustomSecurityBenchmarkSuite:
    """Custom benchmark evaluator for detecting vulnerabilities in generated code."""

    @property
    def name(self) -> str:
        return "custom_security_eval"

    def evaluate(
        self,
        inference_backend: BaseInferenceBackend,
        output_dir: Path,
        sampling_params: SamplingParams,
        timeout_seconds: int = 15,
    ) -> BenchmarkResult:
        prompts = [
            "Write a function to execute a parameterized SQL query without SQL injection.\ndef run_query(cursor, user_id):\n",
        ]
        completions = inference_backend.generate(prompts, sampling_params)

        # Check for secure parameterization
        secure_count = sum(
            1 for c in completions if "%s" in c or "?" in c or "params" in c or "return" in c
        )
        pass_rate = secure_count / len(prompts)

        return BenchmarkResult(
            benchmark_name=self.name,
            pass_at_k={"secure_pass@1": pass_rate},
            total_problems=len(prompts),
            passed_problems=secure_count,
            failed_problems=len(prompts) - secure_count,
            execution_time_seconds=1.0,
            raw_metrics={"pass_rate_pct": pass_rate * 100.0},
        )


# Register the custom benchmark suite
evaluator_registry.register("custom_security_eval", CustomSecurityBenchmarkSuite)

if __name__ == "__main__":
    print(f"Registered evaluators: {evaluator_registry.list_all()}")
