"""
Unit tests for AST Alpha-Equivalence Code Normalizer and Contamination Shield.
"""

from viforge.preprocessing.ast_normalizer import ASTAlphaNormalizer
from viforge.preprocessing.contamination import ContaminationDetector


def test_ast_alpha_renaming():
    """Test that functions with renamed variables and parameters normalize to identical code."""
    code_1 = """
def solve_problem(a, b):
    \"\"\"This is a docstring.\"\"\"
    temp_val = a + b
    return temp_val * 2
"""

    code_2 = """
def custom_solver(x, y):
    # Altered comment
    z = x + y
    return z * 2
"""

    norm_1 = ASTAlphaNormalizer.normalize_code(code_1)
    norm_2 = ASTAlphaNormalizer.normalize_code(code_2)

    assert norm_1 == norm_2
    assert "This is a docstring" not in norm_1
    assert "v_0" in norm_1


def test_ast_alpha_equivalence_check():
    """Test is_alpha_equivalent method across equivalent and non-equivalent snippets."""
    code_a = "def calculate(n):\n    return [i * 2 for i in range(n)]"
    code_b = "def compute(count):\n    return [item * 2 for item in range(count)]"
    code_c = "def compute(count):\n    return [item * 3 for item in range(count)]"

    assert ASTAlphaNormalizer.is_alpha_equivalent(code_a, code_b) is True
    assert ASTAlphaNormalizer.is_alpha_equivalent(code_a, code_c) is False


def test_ast_hash_consistency():
    """Test that alpha-equivalent code produces identical SHA-256 hashes."""
    code_1 = "def process(val):\n    result = val ** 2\n    return result"
    code_2 = "def execute(x):\n    output = x ** 2\n    return output"

    hash_1 = ASTAlphaNormalizer.compute_ast_hash(code_1)
    hash_2 = ASTAlphaNormalizer.compute_ast_hash(code_2)

    assert len(hash_1) == 64
    assert hash_1 == hash_2


def test_extract_code_blocks():
    """Test extracting code from markdown fenced blocks and plain strings."""
    markdown_text = """
Here is the solution:
```python
def answer():
    return 42
```
And another variant:
```
def fallback():
    return 0
```
"""
    blocks = ASTAlphaNormalizer.extract_code_blocks(markdown_text)
    assert len(blocks) == 2
    assert "def answer():" in blocks[0]
    assert "def fallback():" in blocks[1]

    plain_code = "def plain(): return True"
    plain_blocks = ASTAlphaNormalizer.extract_code_blocks(plain_code)
    assert len(plain_blocks) == 1
    assert plain_blocks[0] == plain_code


def test_fallback_on_invalid_syntax():
    """Test graceful fallback when encountering non-Python or syntax-error snippets."""
    bad_code = "def broken( { return # comment\n    invalid !! code"
    norm = ASTAlphaNormalizer.normalize_code(bad_code)
    assert norm is not None
    assert "# comment" not in norm


def test_contamination_detector_ast_shield():
    """Test that ContaminationDetector detects code leaks with renamed variables using AST shield."""
    detector = ContaminationDetector(ngram_size=15, max_allowed_overlap=0.1, enable_ast_shield=True)

    benchmark_code = """
def canonical_fibonacci(n):
    \"\"\"Reference benchmark test solution.\"\"\"
    if n <= 1:
        return n
    return canonical_fibonacci(n - 1) + canonical_fibonacci(n - 2)
"""
    detector.register_benchmark_corpus("humaneval_ref", [benchmark_code])

    # Leaked candidate with renamed identifiers and altered comments
    leaked_candidate = """
def my_custom_calc(num):
    # altered comment
    if num <= 1:
        return num
    return my_custom_calc(num - 1) + my_custom_calc(num - 2)
"""
    overlaps = detector.check_sample(leaked_candidate)
    assert overlaps["humaneval_ref"] == 1.0

    # Test filter_dataset drops the AST-equivalent leaked sample
    dataset = [
        {"id": 1, "text": leaked_candidate},
        {"id": 2, "text": "def independent_logic(a, b): return a / (b + 1e-5)"},
    ]

    clean_records, stats = detector.filter_dataset(dataset)
    assert len(clean_records) == 1
    assert clean_records[0]["id"] == 2
    assert stats["contaminated_records_removed"] == 1
