"""
Unit tests for GRPO reasoning trace extraction, syntax rewards, and length penalty regularization.
"""

import pytest
from viforge.methods.preference import compute_grpo_reward, extract_reasoning_and_code
from viforge.security.sandbox import ExecutionSandbox


def test_extract_reasoning_and_code_with_think_tags():
    raw_text = (
        "<think>\n"
        "We need to implement a function that calculates the nth Fibonacci number efficiently.\n"
        "Using dynamic programming or memoization avoids exponential recursion.\n"
        "</think>\n"
        "```python\n"
        "def fibonacci(n: int) -> int:\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    a, b = 0, 1\n"
        "    for _ in range(2, n + 1):\n"
        "        a, b = b, a + b\n"
        "    return b\n"
        "```"
    )
    reasoning, code = extract_reasoning_and_code(raw_text)
    assert reasoning is not None
    assert "calculates the nth Fibonacci" in reasoning
    assert "def fibonacci(n: int) -> int:" in code
    assert "<think>" not in code
    assert "```" not in code


def test_extract_reasoning_and_code_plain():
    raw_text = "def hello():\n    return 'world'\n"
    reasoning, code = extract_reasoning_and_code(raw_text)
    assert reasoning is None
    assert code == raw_text.strip()


def test_compute_grpo_reward_invalid_syntax():
    reward = compute_grpo_reward("def invalid_syntax(x: return")
    assert reward == 0.0


def test_compute_grpo_reward_valid_syntax_no_tests():
    reward = compute_grpo_reward("def valid_func():\n    return 42\n")
    assert reward == 1.0


def test_compute_grpo_reward_with_reasoning_bonus():
    completion = (
        "<think>\n"
        "First verify that input is positive, then return factorial recursively.\n"
        "</think>\n"
        "def fact(n):\n"
        "    return 1 if n <= 1 else n * fact(n - 1)\n"
    )
    reward = compute_grpo_reward(completion)
    # 1.0 base + 0.1 reasoning bonus = 1.1
    assert reward == pytest.approx(1.1, abs=0.01)


def test_compute_grpo_reward_with_sandbox_tests():
    sandbox = ExecutionSandbox(default_timeout_sec=5)
    valid_code = "def add(a, b):\n    return a + b\n"
    passing_tests = "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n"
    failing_tests = "assert add(2, 3) == 999\n"

    r_pass = compute_grpo_reward(valid_code, test_cases=passing_tests, sandbox=sandbox)
    assert r_pass == 1.0

    r_fail = compute_grpo_reward(valid_code, test_cases=failing_tests, sandbox=sandbox)
    assert r_fail == 0.3


def test_compute_grpo_reward_length_penalty():
    # Generate completion that exceeds soft limit
    long_code = "def large_func():\n" + ("    x = 1\n" * 500)
    reward_normal = compute_grpo_reward(long_code, max_length_soft_cap=20000)
    reward_penalized = compute_grpo_reward(long_code, max_length_soft_cap=500)
    assert reward_penalized < reward_normal
