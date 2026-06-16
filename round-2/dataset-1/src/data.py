#!/usr/bin/env python3
"""Load QQP and PAWS-Wiki datasets and standardize to exp_sel_data_out.json schema."""

# /// script
# requires-python = ">=3.12"
# dependencies = ["loguru"]
# ///

import json
import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WORKSPACE = Path(__file__).parent
DATASET_DIR = WORKSPACE / "temp" / "datasets"
OUTPUT = WORKSPACE / "full_data_out.json"

LABEL_MAP = {0: "not_duplicate", 1: "duplicate"}


def word_count(s: str) -> int:
    return len(s.split())


def length_stratum(text1: str, text2: str) -> str:
    avg = (word_count(text1) + word_count(text2)) / 2
    if avg <= 6:
        return "short"
    elif avg <= 15:
        return "medium"
    else:
        return "long"


@logger.catch(reraise=True)
def load_paws() -> list[dict]:
    path = DATASET_DIR / "full_google-research-datasets_paws_labeled_final_test.json"
    logger.info(f"Loading PAWS-Wiki from {path}")
    rows = json.loads(path.read_text())
    examples = []
    for row in rows:
        s1 = str(row["sentence1"])
        s2 = str(row["sentence2"])
        label = int(row["label"])
        examples.append({
            "input": json.dumps({"sentence1": s1, "sentence2": s2}),
            "output": LABEL_MAP[label],
            "metadata_label_int": label,
            "metadata_stratum": length_stratum(s1, s2),
            "metadata_word_count_s1": word_count(s1),
            "metadata_word_count_s2": word_count(s2),
            "metadata_task_type": "binary_classification",
            "metadata_source_id": int(row["id"]),
        })
    logger.info(f"PAWS-Wiki: {len(examples)} examples")
    return examples


@logger.catch(reraise=True)
def load_qqp() -> list[dict]:
    path = DATASET_DIR / "full_nyu-mll_glue_qqp_validation.json"
    logger.info(f"Loading GLUE QQP from {path}")
    rows = json.loads(path.read_text())
    examples = []
    for i, row in enumerate(rows):
        q1 = str(row["question1"])
        q2 = str(row["question2"])
        label = int(row["label"])
        examples.append({
            "input": json.dumps({"question1": q1, "question2": q2}),
            "output": LABEL_MAP[label],
            "metadata_label_int": label,
            "metadata_stratum": length_stratum(q1, q2),
            "metadata_word_count_q1": word_count(q1),
            "metadata_word_count_q2": word_count(q2),
            "metadata_task_type": "binary_classification",
            "metadata_row_index": i,
        })
    logger.info(f"GLUE QQP: {len(examples)} examples")
    return examples


@logger.catch(reraise=True)
def main() -> None:
    paws = load_paws()
    qqp = load_qqp()

    result = {
        "metadata": {
            "description": "Paraphrase/near-duplicate detection datasets for binary classification evaluation",
            "task": "near_duplicate_detection",
            "label_mapping": LABEL_MAP,
            "datasets_selected": ["paws_wiki_test", "glue_qqp_validation"],
            "selection_rationale": (
                "QQP provides 40k short question pairs with diverse length strata ideal for near-duplicate short text evaluation. "
                "PAWS-Wiki provides 8k adversarial high-lexical-overlap pairs from Wikipedia for cross-domain challenging evaluation."
            ),
        },
        "datasets": [
            {"dataset": "glue_qqp_validation", "examples": qqp},
            {"dataset": "paws_wiki_test", "examples": paws},
        ],
    }

    OUTPUT.write_text(json.dumps(result, indent=2))
    total = sum(len(d["examples"]) for d in result["datasets"])
    logger.info(f"Saved {total} total examples to {OUTPUT}")
    for d in result["datasets"]:
        logger.info(f"  {d['dataset']}: {len(d['examples'])} examples")


if __name__ == "__main__":
    main()
