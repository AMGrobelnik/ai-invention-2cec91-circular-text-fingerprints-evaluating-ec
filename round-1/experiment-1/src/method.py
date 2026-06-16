#!/usr/bin/env python3
"""Circular Text Fingerprints (CTF) vs MinHash for near-duplicate detection on QQP."""

import gc
import hashlib
import json
import math
import os
import resource
import sys
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split

WORKSPACE = Path(__file__).parent
LOGS_DIR = WORKSPACE / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Hardware ──────────────────────────────────────────────────────────────────
NUM_CPUS = min(6, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)
TOTAL_RAM_GB = 43.0
RAM_BUDGET = int(30 * 1024**3)  # 30GB safe limit
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET, RAM_BUDGET))

SEED = 42
N_PAIRS = 10_000
N_HASHES = 128


# ── Hashing helpers ───────────────────────────────────────────────────────────
def _stable_hash(val: Any) -> int:
    """Stable, seed-independent integer hash via SHA-256."""
    h = hashlib.sha256(str(val).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little")


def _multi_hash(val: Any, seed: int) -> int:
    """Hash with seed for MinHash."""
    h = hashlib.sha256(f"{seed}:{val}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little")


# ── MinHash baseline ──────────────────────────────────────────────────────────
def make_shingles(text: str, k: int = 3) -> set[tuple]:
    tokens = text.lower().split()
    if len(tokens) < k:
        return set(tuple(tokens))  # fallback to unigrams if too short
    return {tuple(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def minhash_signature(shingles: set, num_hashes: int = N_HASHES) -> np.ndarray:
    sig = np.full(num_hashes, np.iinfo(np.uint64).max, dtype=np.uint64)
    for shingle in shingles:
        key = str(shingle)
        for seed in range(num_hashes):
            h = _multi_hash(key, seed) & 0xFFFFFFFFFFFFFFFF
            if h < sig[seed]:
                sig[seed] = h
    return sig


def jaccard_minhash(sig1: np.ndarray, sig2: np.ndarray) -> float:
    return float(np.mean(sig1 == sig2))


# ── Circular Text Fingerprints ────────────────────────────────────────────────
def ctf_fingerprint(text: str, radius: int = 2) -> frozenset[int]:
    tokens = text.lower().split()
    if not tokens:
        return frozenset()
    n = len(tokens)
    current = [_stable_hash(t) for t in tokens]
    features: set[int] = set(current)

    for _ in range(radius):
        nxt = []
        for i in range(n):
            neighbors = sorted(
                current[j] for j in range(max(0, i - 1), min(n, i + 2)) if j != i
            )
            new_feat = _stable_hash((current[i], tuple(neighbors))) & 0xFFFFFFFFFFFFFFFF
            nxt.append(new_feat)
            features.add(new_feat)
        current = nxt

    return frozenset(features)


def jaccard_sets(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union


# ── Data loading ──────────────────────────────────────────────────────────────
def load_qqp() -> list[dict]:
    logger.info("Loading QQP dataset from HuggingFace (nyu-mll/glue)…")
    from datasets import load_dataset  # type: ignore
    ds = load_dataset("nyu-mll/glue", "qqp", split="train")
    logger.info(f"Loaded {len(ds)} raw QQP examples")
    pairs = []
    for row in ds:
        q1 = str(row["question1"]).strip()
        q2 = str(row["question2"]).strip()
        label = int(row["label"])
        if len(q1) >= 3 and len(q2) >= 3:
            pairs.append({"q1": q1, "q2": q2, "label": label})
    logger.info(f"Filtered pairs: {len(pairs)}")
    return pairs


def sample_balanced(pairs: list[dict], n: int, rng: np.random.Generator) -> list[dict]:
    pos = [p for p in pairs if p["label"] == 1]
    neg = [p for p in pairs if p["label"] == 0]
    half = n // 2
    pos_idx = rng.choice(len(pos), size=min(half, len(pos)), replace=False)
    neg_idx = rng.choice(len(neg), size=min(half, len(neg)), replace=False)
    sample = [pos[i] for i in pos_idx] + [neg[i] for i in neg_idx]
    rng.shuffle(sample)
    return sample


# ── Threshold sweep ───────────────────────────────────────────────────────────
def sweep_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.0, 1.01, 0.01):
        preds = (scores >= thr).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(scores: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    preds = (scores >= thr).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "count": len(labels)}


def stratify_evaluate(
    scores: np.ndarray, labels: np.ndarray, lengths: np.ndarray, thr: float
) -> dict:
    short_mask = lengths < 40
    mid_mask = (lengths >= 40) & (lengths <= 100)
    long_mask = lengths > 100
    result = {"overall": evaluate(scores, labels, thr)}
    for name, mask in [("short", short_mask), ("medium", mid_mask), ("long", long_mask)]:
        if mask.sum() > 0:
            result[name] = evaluate(scores[mask], labels[mask], thr)
        else:
            result[name] = {"count": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
@logger.catch(reraise=True)
def main() -> None:
    rng = np.random.default_rng(SEED)

    # ── Load data ────────────────────────────────────────────────────────────
    all_pairs = load_qqp()
    logger.info(f"Total filtered pairs: {len(all_pairs)}")
    sample = sample_balanced(all_pairs, N_PAIRS, rng)
    del all_pairs
    gc.collect()
    logger.info(f"Balanced sample: {len(sample)} pairs ({sum(p['label'] for p in sample)} duplicates)")

    labels = np.array([p["label"] for p in sample])
    lengths = np.array([len(p["q1"]) + len(p["q2"]) for p in sample])

    # Split: 60/20/20
    idx = np.arange(len(sample))
    train_idx, tmp_idx = train_test_split(idx, test_size=0.4, random_state=SEED, stratify=labels)
    val_idx, test_idx = train_test_split(tmp_idx, test_size=0.5, random_state=SEED, stratify=labels[tmp_idx])
    logger.info(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_val_idx = np.concatenate([train_idx, val_idx])

    # ── Compute MinHash scores ───────────────────────────────────────────────
    logger.info("Computing MinHash signatures…")
    mh_scores = np.zeros(len(sample))
    for i, p in enumerate(sample):
        s1 = make_shingles(p["q1"])
        s2 = make_shingles(p["q2"])
        if not s1 and not s2:
            mh_scores[i] = 1.0
        elif not s1 or not s2:
            mh_scores[i] = 0.0
        else:
            sig1 = minhash_signature(s1)
            sig2 = minhash_signature(s2)
            mh_scores[i] = jaccard_minhash(sig1, sig2)
        if (i + 1) % 1000 == 0:
            logger.info(f"  MinHash: {i + 1}/{len(sample)}")
    logger.info("MinHash done")

    # ── Compute CTF scores ───────────────────────────────────────────────────
    ctf_scores: dict[int, np.ndarray] = {}
    for radius in [1, 2, 3]:
        logger.info(f"Computing CTF fingerprints (radius={radius})…")
        scores = np.zeros(len(sample))
        for i, p in enumerate(sample):
            fp1 = ctf_fingerprint(p["q1"], radius=radius)
            fp2 = ctf_fingerprint(p["q2"], radius=radius)
            scores[i] = jaccard_sets(fp1, fp2)
            if (i + 1) % 1000 == 0:
                logger.info(f"  CTF R={radius}: {i + 1}/{len(sample)}")
        ctf_scores[radius] = scores
        logger.info(f"CTF R={radius} done")

    # ── Threshold optimization on train+val ──────────────────────────────────
    logger.info("Optimizing thresholds on train+val…")
    tv_labels = labels[train_val_idx]
    tv_lengths = lengths[train_val_idx]

    mh_thr, mh_tv_f1 = sweep_threshold(mh_scores[train_val_idx], tv_labels)
    logger.info(f"MinHash best threshold={mh_thr:.2f}, train+val F1={mh_tv_f1:.4f}")

    ctf_thresholds: dict[int, float] = {}
    for radius in [1, 2, 3]:
        thr, tv_f1 = sweep_threshold(ctf_scores[radius][train_val_idx], tv_labels)
        ctf_thresholds[radius] = thr
        logger.info(f"CTF R={radius} best threshold={thr:.2f}, train+val F1={tv_f1:.4f}")

    # ── Test evaluation ──────────────────────────────────────────────────────
    test_labels = labels[test_idx]
    test_lengths = lengths[test_idx]

    logger.info("Evaluating on test set…")
    mh_results = stratify_evaluate(mh_scores[test_idx], test_labels, test_lengths, mh_thr)
    ctf_results: dict[str, dict] = {}
    for radius in [1, 2, 3]:
        ctf_results[f"radius_{radius}"] = stratify_evaluate(
            ctf_scores[radius][test_idx], test_labels, test_lengths, ctf_thresholds[radius]
        )

    # ── Success criterion ────────────────────────────────────────────────────
    mh_f1 = mh_results["overall"]["f1"]
    ctf_best_f1 = max(ctf_results[f"radius_{r}"]["overall"]["f1"] for r in [1, 2, 3])
    ctf_best_r = max([1, 2, 3], key=lambda r: ctf_results[f"radius_{r}"]["overall"]["f1"])
    delta = ctf_best_f1 - mh_f1

    ctf_short_f1 = ctf_results[f"radius_{ctf_best_r}"].get("short", {}).get("f1", 0.0)
    mh_short_f1 = mh_results.get("short", {}).get("f1", 0.0)
    short_delta = ctf_short_f1 - mh_short_f1

    if delta >= 0.03:
        success = True
        criterion_met = "full"
        finding = f"CTF (R={ctf_best_r}) outperforms MinHash by {delta:.4f} F1 points overall (>= 0.03 threshold). Hypothesis SUPPORTED."
    elif short_delta >= 0.02:
        success = True
        criterion_met = "partial"
        finding = f"CTF (R={ctf_best_r}) outperforms MinHash by {short_delta:.4f} F1 points on short texts (< 40 chars). Partial hypothesis support."
    else:
        success = False
        criterion_met = "none"
        finding = f"CTF does not outperform MinHash by the required margin. Overall delta={delta:.4f}, short-text delta={short_delta:.4f}. Hypothesis NOT supported — iterative neighborhood hashing may add noise for text vs. molecules."

    logger.info(f"Result: success={success}, delta={delta:.4f}")
    logger.info(f"Finding: {finding}")

    # ── Log comparison table ─────────────────────────────────────────────────
    logger.info("Results table:")
    logger.info(f"{'Method':<14} {'R':>2} {'Thr':>5} {'Overall F1':>10} {'Short F1':>9} {'Med F1':>7} {'Long F1':>8}")
    logger.info(f"{'MinHash':<14} {'N/A':>2} {mh_thr:>5.2f} {mh_f1:>10.4f} {mh_results.get('short', {}).get('f1', 0):>9.4f} {mh_results.get('medium', {}).get('f1', 0):>7.4f} {mh_results.get('long', {}).get('f1', 0):>8.4f}")
    for r in [1, 2, 3]:
        cr = ctf_results[f"radius_{r}"]
        logger.info(f"{'CTF':<14} {r:>2} {ctf_thresholds[r]:>5.2f} {cr['overall']['f1']:>10.4f} {cr.get('short', {}).get('f1', 0):>9.4f} {cr.get('medium', {}).get('f1', 0):>7.4f} {cr.get('long', {}).get('f1', 0):>8.4f}")

    # ── Build method_out.json (exp_gen_sol_out schema) ────────────────────────
    logger.info("Building method_out.json…")
    examples = []
    for i in test_idx:
        p = sample[i]
        char_len = len(p["q1"]) + len(p["q2"])
        if char_len < 40:
            length_bucket = "short"
        elif char_len <= 100:
            length_bucket = "medium"
        else:
            length_bucket = "long"

        label_str = "duplicate" if p["label"] == 1 else "not_duplicate"
        mh_pred = "duplicate" if mh_scores[i] >= mh_thr else "not_duplicate"
        ctf_preds = {
            r: ("duplicate" if ctf_scores[r][i] >= ctf_thresholds[r] else "not_duplicate")
            for r in [1, 2, 3]
        }

        examples.append({
            "input": f"Question 1: {p['q1']}\nQuestion 2: {p['q2']}",
            "output": label_str,
            "predict_minhash": mh_pred,
            "predict_ctf_r1": ctf_preds[1],
            "predict_ctf_r2": ctf_preds[2],
            "predict_ctf_r3": ctf_preds[3],
            "metadata_minhash_score": round(float(mh_scores[i]), 4),
            "metadata_ctf_r1_score": round(float(ctf_scores[1][i]), 4),
            "metadata_ctf_r2_score": round(float(ctf_scores[2][i]), 4),
            "metadata_ctf_r3_score": round(float(ctf_scores[3][i]), 4),
            "metadata_char_length": char_len,
            "metadata_length_bucket": length_bucket,
        })

    method_out = {
        "metadata": {
            "method_name": "Circular Text Fingerprints (CTF) vs MinHash",
            "description": "Near-duplicate detection on QQP using ECFP-inspired token neighborhood hashing vs MinHash 3-gram shingling",
            "parameters": {
                "n_pairs_sampled": N_PAIRS,
                "n_minhash_functions": N_HASHES,
                "ctf_radii": [1, 2, 3],
                "ctf_window": 1,
                "shingle_k": 3,
                "seed": SEED,
            },
            "success": success,
            "criterion_met": criterion_met,
            "criterion": "CTF F1 >= MinHash F1 + 0.03 overall, or +0.02 in short-text stratum (<40 chars)",
            "minhash_baseline_f1": mh_f1,
            "ctf_best_f1": ctf_best_f1,
            "ctf_best_radius": ctf_best_r,
            "improvement_f1_points": round(delta, 4),
            "findings": finding,
            "thresholds": {
                "minhash": mh_thr,
                **{f"ctf_r{r}": ctf_thresholds[r] for r in [1, 2, 3]},
            },
            "results_by_method": {
                "minhash": mh_results,
                "ctf": ctf_results,
            },
            "split_sizes": {
                "train": len(train_idx),
                "val": len(val_idx),
                "test": len(test_idx),
            },
        },
        "datasets": [
            {
                "dataset": "quora_question_pairs",
                "examples": examples,
            }
        ],
    }

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(method_out, indent=2))
    logger.info(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB, {len(examples)} examples)")


if __name__ == "__main__":
    main()
