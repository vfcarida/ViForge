"""
ViForge Standalone Dataset Decontamination & Deduplication Script.
"""

import sys
from pathlib import Path
from viforge.datasets.adapters import DatasetAdapter
from viforge.preprocessing.deduplication import MinHashDeduplicator
from viforge.preprocessing.contamination import ContaminationDetector


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python decontaminate_dataset.py <input_dataset.jsonl> <output_dataset.parquet>"
        )
        return

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    print(f"Loading records from {in_path}...")
    records = DatasetAdapter.load_local_records(in_path)

    print("Running MinHash LSH deduplication...")
    dedup = MinHashDeduplicator()
    deduped_records, dedup_stats = dedup.deduplicate_corpus(records)

    print("Running 10-gram benchmark contamination detector...")
    detector = ContaminationDetector()
    clean_records, contam_stats = detector.filter_dataset(deduped_records)

    print(f"Saving {len(clean_records)} clean records to {out_path}...")
    DatasetAdapter.save_records(clean_records, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
