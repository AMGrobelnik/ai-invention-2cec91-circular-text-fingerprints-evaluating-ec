#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["loguru"]
# ///
"""Load QQP and PAWS datasets, standardize to exp_sel_data_out schema."""

import json
import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/data.log", rotation="30 MB", level="DEBUG")

WORKSPACE = Path(__file__).parent
DATASETS_DIR = WORKSPACE / "temp" / "datasets"
OUT_PATH = WORKSPACE / "full_data_out.json"


def load_qqp() -> list[dict]:
    """Load Quora Question Pairs (pair-class, train) from temp/datasets."""
    path = DATASETS_DIR / "full_sentence-transformers_quora-duplicates_pair-class_train.json"
    logger.info(f"Loading QQP from {path}")
    rows = json.loads(path.read_text())
    logger.info(f"QQP raw rows: {len(rows)}")

    examples = []
    for i, r in enumerate(rows):
        s1 = str(r["sentence1"]).strip()
        s2 = str(r["sentence2"]).strip()
        label = int(r["label"])
        inp = json.dumps({"text1": s1, "text2": s2}, ensure_ascii=False)
        examples.append({
            "input": inp,
            "output": str(label),
            "metadata_is_duplicate": label,
            "metadata_word_count_1": len(s1.split()),
            "metadata_word_count_2": len(s2.split()),
            "metadata_char_count_1": len(s1),
            "metadata_char_count_2": len(s2),
            "metadata_row_index": i,
            "metadata_task_type": "binary_classification",
            "metadata_n_classes": 2,
        })

    logger.info(f"QQP examples built: {len(examples)}")
    return examples


def load_paws() -> list[dict]:
    """Load PAWS labeled_final train from temp/datasets."""
    path = DATASETS_DIR / "full_paws_labeled_final_train.json"
    logger.info(f"Loading PAWS from {path}")
    rows = json.loads(path.read_text())
    logger.info(f"PAWS raw rows: {len(rows)}")

    examples = []
    for i, r in enumerate(rows):
        s1 = str(r["sentence1"]).strip()
        s2 = str(r["sentence2"]).strip()
        label = int(r["label"])
        inp = json.dumps({"text1": s1, "text2": s2}, ensure_ascii=False)
        examples.append({
            "input": inp,
            "output": str(label),
            "metadata_is_duplicate": label,
            "metadata_word_count_1": len(s1.split()),
            "metadata_word_count_2": len(s2.split()),
            "metadata_char_count_1": len(s1),
            "metadata_char_count_2": len(s2),
            "metadata_row_index": i,
            "metadata_task_type": "binary_classification",
            "metadata_n_classes": 2,
        })

    logger.info(f"PAWS examples built: {len(examples)}")
    return examples


@logger.catch(reraise=True)
def main() -> None:
    Path("logs").mkdir(exist_ok=True)

    qqp_examples = load_qqp()

    output = {
        "metadata": {
            "description": "Quora Question Pairs for near-duplicate detection: CTF vs MinHash evaluation",
            "source": "HuggingFace Hub — sentence-transformers/quora-duplicates (pair-class)",
            "total_examples": len(qqp_examples),
            "duplicate_count": sum(1 for e in qqp_examples if e["metadata_is_duplicate"] == 1),
            "non_duplicate_count": sum(1 for e in qqp_examples if e["metadata_is_duplicate"] == 0),
        },
        "datasets": [
            {
                "dataset": "sentence-transformers/quora-duplicates (pair-class)",
                "examples": qqp_examples,
            },
        ],
    }

    OUT_PATH.write_text(json.dumps(output, ensure_ascii=False))
    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    logger.info(f"Wrote {OUT_PATH} ({size_mb:.1f} MB)")
    logger.info(f"QQP: {len(qqp_examples)} examples")


if __name__ == "__main__":
    main()
