"""
ViForge Statistical Significance and Confidence Interval Calculations.
"""

import math
from typing import Optional, Tuple


def wilson_score_interval(
    successes: int, total: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Computes Wilson score confidence interval for binomial pass rates.
    """
    if total == 0:
        return (0.0, 0.0)

    z = 1.95996  # 95% confidence z-score
    if confidence == 0.99:
        z = 2.57583
    elif confidence == 0.90:
        z = 1.64485

    p_hat = successes / total
    denominator = 1 + (z**2 / total)
    centre_adjusted_probability = p_hat + (z**2 / (2 * total))
    adjusted_std_dev = math.sqrt((p_hat * (1 - p_hat) / total) + (z**2 / (4 * (total**2))))

    lower_bound = (centre_adjusted_probability - z * adjusted_std_dev) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_std_dev) / denominator

    return (max(0.0, round(lower_bound, 4)), min(1.0, round(upper_bound, 4)))


def compute_relative_delta(baseline: float, specialized: float) -> float:
    """Computes relative percentage change: ((spec - base) / base) * 100%."""
    if baseline == 0.0:
        return 100.0 if specialized > 0.0 else 0.0
    return round(((specialized - baseline) / baseline) * 100.0, 2)


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Computes the unbiased estimator of pass@k as defined by Chen et al. (HumanEval):
    pass@k = 1 - (comb(n - c, k) / comb(n, k)) = 1 - prod_{i=1}^k ((n - c - i + 1) / (n - i + 1))

    Args:
        n: Total number of generated samples per problem (n >= 1).
        c: Number of correct/passing samples (0 <= c <= n).
        k: Evaluated k threshold (1 <= k <= n).

    Returns:
        float: Estimated pass@k probability in range [0.0, 1.0].
    """
    if n < k:
        raise ValueError(f"Total samples n ({n}) must be greater than or equal to k ({k}).")
    if c < 0 or c > n:
        raise ValueError(f"Correct samples c ({c}) must be between 0 and n ({n}).")
    if k <= 0:
        raise ValueError(f"k ({k}) must be positive integer.")

    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0

    # Unbiased combinatorial product
    prob_all_wrong = 1.0
    for i in range(1, k + 1):
        prob_all_wrong *= (n - c - i + 1) / (n - i + 1)

    return max(0.0, min(1.0, round(1.0 - prob_all_wrong, 4)))


def compute_pass_at_k_distribution(
    problem_results: list[list[bool]], k_list: list[int]
) -> dict[str, float]:
    """
    Computes average pass@k across a collection of problems, where each problem
    contains a list of boolean evaluation results from n generated samples.

    Args:
        problem_results: List of problems, where each problem is list of sample pass booleans [True, False, ...].
        k_list: List of k values to compute (e.g. [1, 5, 10]).

    Returns:
        Dict of metric names to average pass rate, e.g. {'pass@1': 0.68, 'pass@5': 0.85}.
    """
    if not problem_results:
        return {f"pass@{k}": 0.0 for k in k_list}

    metrics: dict[str, float] = {}
    for k in k_list:
        scores = []
        for problem_samples in problem_results:
            n = len(problem_samples)
            c = sum(1 for passed in problem_samples if passed)
            if n >= k:
                scores.append(estimate_pass_at_k(n, c, k))
        avg_k = sum(scores) / max(1, len(scores)) if scores else 0.0
        metrics[f"pass@{k}"] = round(avg_k, 4)

    return metrics


def mcnemar_test(b: int, c: int, exact: Optional[bool] = None) -> Tuple[float, float]:
    """
    Computes McNemar's test for paired nominal trial outcomes.
    b: Number of discordant problems where baseline succeeded and specialist failed.
    c: Number of discordant problems where baseline failed and specialist succeeded.

    Returns:
        (statistic, p_value)
    """
    total_discordant = b + c
    if total_discordant == 0:
        return (0.0, 1.0)

    use_exact = exact if exact is not None else (total_discordant < 25)

    if use_exact:
        k = min(b, c)
        p_val = sum(math.comb(total_discordant, i) * (0.5**total_discordant) for i in range(k + 1))
        p_val = min(1.0, 2.0 * p_val)
        stat = float((b - c) ** 2) / total_discordant
        return (round(stat, 4), round(p_val, 6))
    else:
        num = (abs(b - c) - 1.0) ** 2
        chi2 = max(0.0, num / total_discordant)
        z = math.sqrt(chi2)
        p_val = math.erfc(z / math.sqrt(2.0))
        return (round(chi2, 4), round(min(1.0, max(0.0, p_val)), 6))


def two_proportion_z_test(
    success1: int, total1: int, success2: int, total2: int
) -> Tuple[float, float]:
    """
    Two-tailed pooled two-proportion z-test for independent binomial samples.
    """
    if total1 <= 0 or total2 <= 0:
        return (0.0, 1.0)

    p1 = success1 / total1
    p2 = success2 / total2
    p_pool = (success1 + success2) / (total1 + total2)

    if p_pool == 0.0 or p_pool == 1.0:
        return (0.0, 1.0)

    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / total1 + 1.0 / total2))
    if se == 0.0:
        return (0.0, 1.0)

    z = abs(p2 - p1) / se
    p_val = math.erfc(z / math.sqrt(2.0))
    return (round(z, 4), round(min(1.0, max(0.0, p_val)), 6))

