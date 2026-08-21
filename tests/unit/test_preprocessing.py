"""
Unit tests for ViForge Preprocessing: Normalizer, MinHash Deduplication, Contamination, and Sequence Packing.
"""

import pytest

from viforge.preprocessing.contamination import ContaminationDetector
from viforge.preprocessing.deduplication import MinHashDeduplicator
from viforge.preprocessing.normalizer import CodeNormalizer
from viforge.preprocessing.packing import SequencePacker


@pytest.mark.unit
def test_code_normalizer():
    raw_code = 'def foo():\n    """Docstring to strip."""\n    # inline comment\n    x = 10\n    return x\n'
    clean = CodeNormalizer.strip_comments(raw_code)
    assert "Docstring to strip" not in clean
    assert "inline comment" not in clean
    assert "return x" in clean

    is_valid, _ = CodeNormalizer.validate_ast(clean)
    assert is_valid is True

    is_bad, err = CodeNormalizer.validate_ast("def broken_syntax(")
    assert is_bad is False
    assert "SyntaxError" in err


@pytest.mark.unit
def test_minhash_deduplication():
    records = [
        {"id": 1, "text": "def quick_sort(arr):\n    if len(arr) <= 1: return arr\n    return arr"},
        {"id": 2, "text": "def quick_sort(arr):\n    if len(arr) <= 1: return arr\n    return arr"},
        {"id": 3, "text": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n): pass"},
    ]
    dedup = MinHashDeduplicator(num_perm=64, jaccard_threshold=0.8)
    clean, stats = dedup.deduplicate_corpus(records)
    assert len(clean) == 2
    assert stats["exact_duplicates_removed"] >= 1


@pytest.mark.unit
def test_contamination_detector():
    detector = ContaminationDetector(ngram_size=5, max_allowed_overlap=0.10)
    detector.register_benchmark_corpus(
        "humaneval",
        [
            "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    for i in numbers: pass"
        ],
    )

    clean_sample = "def calculate_compound_interest(principal, rate, time):\n    return principal * (1 + rate)**time"
    overlaps_clean = detector.check_sample(clean_sample)
    assert overlaps_clean["humaneval"] == 0.0

    leaked_sample = "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    for i in numbers: pass"
    overlaps_leaked = detector.check_sample(leaked_sample)
    assert overlaps_leaked["humaneval"] > 0.5

    # Parameter validation checks
    with pytest.raises(ValueError, match="ngram_size must be positive"):
        ContaminationDetector(ngram_size=0)

    with pytest.raises(ValueError, match="max_allowed_overlap must be in range"):
        ContaminationDetector(max_allowed_overlap=1.5)

    # Filter dataset check
    records = [
        {"id": "clean_1", "text": clean_sample},
        {"id": "leaked_1", "text": leaked_sample},
    ]
    clean_recs, stats = detector.filter_dataset(records)
    assert len(clean_recs) == 1
    assert clean_recs[0]["id"] == "clean_1"
    assert stats["contaminated_records_removed"] == 1


@pytest.mark.unit
def test_sequence_packer():
    packer = SequencePacker(max_seq_len=16, pad_token_id=0, eos_token_id=2)
    seqs = [[10, 11, 12], [20, 21, 22, 23], [30, 31]]
    packed = packer.pack_tokenized_sequences(seqs, add_eos=True)

    assert packed["num_blocks"] >= 1
    assert len(packed["input_ids"][0]) == 16
    assert len(packed["attention_mask"][0]) == 16
