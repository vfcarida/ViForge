"""
ViForge Statistical Significance and Confidence Interval Calculations.
"""

import math
from typing import Tuple


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
