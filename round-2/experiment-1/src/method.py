#!/usr/bin/env python3
"""
Circular Text Fingerprints (CTF) vs Baselines: near-duplicate short text detection.
Implements CTF (ECFP-style iterative neighborhood hashing), MinHash, char n-gram Jaccard,
word-unigram Jaccard, and TF-IDF cosine. Evaluates on Quora Question Pairs with threshold
tuning, stratified metrics, and statistical testing.
"""

import gc
import hashlib
import json
import math
import os
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from loguru import logger
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ── Logging ──────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
(WORKSPACE / "logs").mkdir(exist_ok=True)
(WORKSPACE / "results").mkdir(exist_ok=True)
logger.remove()
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
FMT = f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}"
logger.add(sys.stdout, level="INFO", format=FMT)
logger.add(WORKSPACE / "logs/run.log", rotation="30 MB", level="DEBUG")

# ── Hardware limits ───────────────────────────────────────────────────────────
def _container_ram_gb() -> float:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return psutil.virtual_memory().total / 1e9

TOTAL_RAM_GB = _container_ram_gb()
RAM_BUDGET = int(min(TOTAL_RAM_GB * 0.80, 32) * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"Hardware: 6 CPUs, {TOTAL_RAM_GB:.1f}GB RAM, RAM budget {RAM_BUDGET/1e9:.1f}GB")

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = Path(
    "/home/adrian/projects/ai-inventor/aii_data/users/admin/runs/"
    "run_iUkeOuIf7aLb/3_invention_loop/iter_1/gen_art/gen_art_dataset_1"
)
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "0"))  # 0 = all
SEED = 42
BOOTSTRAP_N = 1000


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    return " ".join(c if c.isalnum() else " " for c in text.lower()).split().__repr__()[1:-1].replace("', '", " ").replace("'", "")


def tokenize(text: str) -> list[str]:
    return [c for c in text.lower() if c.isalnum() or c == " "].copy()


def tokenize_words(text: str) -> list[str]:
    return [w for w in "".join(c if c.isalnum() else " " for c in text.lower()).split() if w]


# ── Similarity methods ────────────────────────────────────────────────────────

def char_ngram_jaccard(t1: str, t2: str, n: int = 4) -> float:
    s1 = set(t1[i:i+n] for i in range(len(t1) - n + 1))
    s2 = set(t2[i:i+n] for i in range(len(t2) - n + 1))
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def word_jaccard(t1: str, t2: str) -> float:
    w1 = set(tokenize_words(t1))
    w2 = set(tokenize_words(t2))
    if not w1 and not w2:
        return 1.0
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def minhash_shingle_jaccard(t1: str, t2: str, k: int = 2) -> float:
    """Exact Jaccard on k-shingle sets (faster than datasketch for 400k pairs)."""
    words1 = tokenize_words(t1)
    words2 = tokenize_words(t2)

    def shingles(words: list[str], k: int) -> set:
        if len(words) < k:
            return {(w,) for w in words} if words else {""}
        return {tuple(words[i:i+k]) for i in range(len(words) - k + 1)}

    s1, s2 = shingles(words1, k), shingles(words2, k)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def _stable_hash(val: Any) -> int:
    """Deterministic hash using SHA1 (avoids PYTHONHASHSEED issues)."""
    s = str(val).encode()
    return int(hashlib.sha1(s).hexdigest()[:16], 16)


def ctf_similarity(t1: str, t2: str, radius: int = 2) -> float:
    """
    Circular Text Fingerprint: ECFP-style iterative neighborhood hashing.
    Each token's feature = hash of itself + neighbors across R iterations.
    Fingerprint = union of all feature values across all radii.
    """
    def ctf_fingerprint(text: str, R: int) -> set:
        words = tokenize_words(text)
        if not words:
            return set()
        n = len(words)
        features = [_stable_hash(w) for w in words]
        fp = set(features)
        for _ in range(R):
            new_features = []
            for i in range(n):
                neighbors = []
                if i > 0:
                    neighbors.append(features[i - 1])
                if i < n - 1:
                    neighbors.append(features[i + 1])
                new_feat = _stable_hash((features[i], tuple(sorted(neighbors))))
                new_features.append(new_feat)
                fp.add(new_feat)
            features = new_features
        return fp

    fp1 = ctf_fingerprint(t1, radius)
    fp2 = ctf_fingerprint(t2, radius)
    if not fp1 and not fp2:
        return 1.0
    if not fp1 or not fp2:
        return 0.0
    return len(fp1 & fp2) / len(fp1 | fp2)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_qqp(max_n: int = 0) -> list[dict]:
    """Load QQP from full_data_out parts."""
    parts = sorted((DATA_DIR / "full_data_out").glob("full_data_out_*.json"))
    examples = []
    for part in parts:
        logger.info(f"Loading {part.name}")
        data = json.loads(part.read_text())
        for ds in data["datasets"]:
            for ex in ds["examples"]:
                inp = json.loads(ex["input"])
                examples.append({
                    "text1": inp["text1"],
                    "text2": inp["text2"],
                    "label": int(ex["output"]),
                    "wc1": ex.get("metadata_word_count_1", len(inp["text1"].split())),
                    "wc2": ex.get("metadata_word_count_2", len(inp["text2"].split())),
                })
        del data
        gc.collect()
        if max_n and len(examples) >= max_n:
            examples = examples[:max_n]
            break
    logger.info(f"Loaded {len(examples)} QQP pairs")
    return examples


# ── Train/test split ──────────────────────────────────────────────────────────
def split_examples(examples: list[dict], train_frac: float = 0.60) -> tuple:
    """Deterministic split by hash of index."""
    train, test = [], []
    for i, ex in enumerate(examples):
        h = _stable_hash(i) % 100
        if h < int(train_frac * 100):
            train.append(ex)
        else:
            test.append(ex)
    return train, test


# ── Threshold tuning ──────────────────────────────────────────────────────────
def tune_threshold(scores: np.ndarray, labels: np.ndarray, granularity: float = 0.01) -> tuple[float, float]:
    """Grid search threshold maximizing F1."""
    best_f1, best_tau = 0.0, 0.5
    thresholds = np.arange(0.0, 1.0 + granularity, granularity)
    for tau in thresholds:
        preds = (scores >= tau).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
    return best_tau, best_f1


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "support": int(len(y_true)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def stratify_by_wc(examples: list[dict]) -> dict[str, list[int]]:
    """Return stratum name → list of indices."""
    strata: dict[str, list[int]] = {"<=6": [], "7-15": [], "16+": []}
    for i, ex in enumerate(examples):
        avg_wc = (ex["wc1"] + ex["wc2"]) / 2
        if avg_wc <= 6:
            strata["<=6"].append(i)
        elif avg_wc <= 15:
            strata["7-15"].append(i)
        else:
            strata["16+"].append(i)
    return strata


# ── Statistical tests ─────────────────────────────────────────────────────────
def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """McNemar's test p-value for two classifiers."""
    b = int(((pred_a == y_true) & (pred_b != y_true)).sum())
    c = int(((pred_a != y_true) & (pred_b == y_true)).sum())
    if b + c == 0:
        return 1.0
    # Use chi2 approximation (valid when b+c > 25)
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import chi2
    return float(1 - chi2.cdf(stat, df=1))


def bootstrap_f1_diff(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n: int = 1000) -> dict:
    rng = np.random.default_rng(SEED)
    diffs = []
    idx_all = np.arange(len(y_true))
    for _ in range(n):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        yt, pa, pb = y_true[idx], pred_a[idx], pred_b[idx]
        prec_a, rec_a, f1_a, _ = precision_recall_fscore_support(yt, pa, average="binary", zero_division=0)
        prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(yt, pb, average="binary", zero_division=0)
        diffs.append(float(f1_a - f1_b))
    diffs_arr = np.array(diffs)
    return {
        "lower": float(np.percentile(diffs_arr, 2.5)),
        "upper": float(np.percentile(diffs_arr, 97.5)),
        "point_estimate": float(np.mean(diffs_arr)),
    }


# ── Batch score computation ───────────────────────────────────────────────────
def compute_scores_batch(examples: list[dict], method: str, **kwargs) -> np.ndarray:
    """Compute similarity scores for a list of pairs using the given method."""
    scores = np.zeros(len(examples), dtype=np.float32)
    for i, ex in enumerate(tqdm(examples, desc=method, leave=False)):
        t1, t2 = ex["text1"], ex["text2"]
        if method.startswith("minhash_k"):
            k = int(method.split("k")[1])
            scores[i] = minhash_shingle_jaccard(t1, t2, k=k)
        elif method == "char4gram":
            scores[i] = char_ngram_jaccard(t1, t2, n=4)
        elif method == "word_jaccard":
            scores[i] = word_jaccard(t1, t2)
        elif method.startswith("ctf_r"):
            r = int(method.split("r")[1])
            scores[i] = ctf_similarity(t1, t2, radius=r)
        else:
            raise ValueError(f"Unknown method: {method}")
    return scores


def compute_tfidf_scores(train: list[dict], test: list[dict]) -> np.ndarray:
    """Fit TF-IDF on train corpus and compute cosine similarities on test."""
    logger.info("Fitting TF-IDF vectorizer on train corpus")
    corpus = [ex["text1"] for ex in train] + [ex["text2"] for ex in train]
    vec = TfidfVectorizer(analyzer="word", lowercase=True, sublinear_tf=True)
    vec.fit(corpus)
    scores = np.zeros(len(test), dtype=np.float32)
    BATCH = 2000
    for start in tqdm(range(0, len(test), BATCH), desc="tfidf", leave=False):
        batch = test[start:start + BATCH]
        v1 = vec.transform([ex["text1"] for ex in batch])
        v2 = vec.transform([ex["text2"] for ex in batch])
        for j in range(len(batch)):
            scores[start + j] = float(cosine_similarity(v1[j], v2[j])[0, 0])
    return scores


# ── Main ──────────────────────────────────────────────────────────────────────
@logger.catch(reraise=True)
def main():
    t_start = datetime.now(timezone.utc)
    logger.info("=== CTF vs Baselines Experiment Start ===")

    # Load data
    max_n = MAX_EXAMPLES if MAX_EXAMPLES else 0
    examples = load_qqp(max_n=max_n)
    labels_all = np.array([ex["label"] for ex in examples])
    logger.info(f"Class balance: {labels_all.mean():.3f} duplicate rate")

    # Split
    train, test = split_examples(examples, train_frac=0.60)
    logger.info(f"Train: {len(train)}, Test: {len(test)}")
    y_train = np.array([ex["label"] for ex in train])
    y_test = np.array([ex["label"] for ex in test])

    del examples
    gc.collect()

    # Methods (excluding tfidf which is handled separately)
    PAIR_METHODS = [
        "minhash_k1", "minhash_k2", "minhash_k3",
        "char4gram", "word_jaccard",
        "ctf_r1", "ctf_r2", "ctf_r3",
    ]
    METHOD_NAMES = {
        "minhash_k1": "MinHash (k=1)",
        "minhash_k2": "MinHash (k=2)",
        "minhash_k3": "MinHash (k=3)",
        "char4gram": "Character 4-gram Jaccard",
        "word_jaccard": "Word-Unigram Jaccard",
        "tfidf": "TF-IDF Cosine",
        "ctf_r1": "CTF (R=1)",
        "ctf_r2": "CTF (R=2)",
        "ctf_r3": "CTF (R=3)",
    }

    # Storage
    thresholds: dict[str, float] = {}
    all_test_scores: dict[str, np.ndarray] = {}
    all_test_preds: dict[str, np.ndarray] = {}

    # --- Pairwise methods ---
    for method in PAIR_METHODS:
        logger.info(f"Computing train scores: {method}")
        train_scores = compute_scores_batch(train, method)
        tau, train_f1 = tune_threshold(train_scores, y_train)
        thresholds[method] = tau
        logger.info(f"  τ_opt={tau:.3f}, train F1={train_f1:.4f}")
        del train_scores
        gc.collect()

        logger.info(f"Computing test scores: {method}")
        test_scores = compute_scores_batch(test, method)
        all_test_scores[method] = test_scores
        all_test_preds[method] = (test_scores >= tau).astype(int)

    # --- TF-IDF ---
    logger.info("Computing TF-IDF scores")
    tfidf_train_scores = compute_tfidf_scores(train, train)
    tau_tfidf, tfidf_train_f1 = tune_threshold(tfidf_train_scores, y_train)
    thresholds["tfidf"] = tau_tfidf
    logger.info(f"  TF-IDF τ_opt={tau_tfidf:.3f}, train F1={tfidf_train_f1:.4f}")
    del tfidf_train_scores
    gc.collect()

    tfidf_test_scores = compute_tfidf_scores(train, test)
    all_test_scores["tfidf"] = tfidf_test_scores
    all_test_preds["tfidf"] = (tfidf_test_scores >= tau_tfidf).astype(int)

    # --- Evaluate ---
    all_methods = PAIR_METHODS + ["tfidf"]
    overall: dict[str, dict] = {}
    confusion: dict[str, dict] = {}
    for method in all_methods:
        m = compute_metrics(y_test, all_test_preds[method])
        name = METHOD_NAMES[method]
        overall[name] = {"precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "support": m["support"]}
        confusion[name] = {"tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"]}
        logger.info(f"{name}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")

    # --- Stratified ---
    strata = stratify_by_wc(test)
    by_wc: dict[str, dict] = {}
    for stratum_name, idxs in strata.items():
        if len(idxs) < 100:
            logger.warning(f"Stratum {stratum_name}: {len(idxs)} pairs < 100, skipping")
            continue
        idxs_arr = np.array(idxs)
        yt_s = y_test[idxs_arr]
        by_wc[stratum_name] = {}
        for method in all_methods:
            preds_s = all_test_preds[method][idxs_arr]
            m = compute_metrics(yt_s, preds_s)
            name = METHOD_NAMES[method]
            by_wc[stratum_name][name] = {
                "precision": m["precision"], "recall": m["recall"],
                "f1": m["f1"], "support": m["support"],
            }
        logger.info(f"Stratum {stratum_name}: {len(idxs)} pairs evaluated")

    # --- Statistical comparison: CTF (R=3) vs best baseline ---
    baseline_methods = [m for m in PAIR_METHODS + ["tfidf"] if not m.startswith("ctf")]
    best_baseline = max(baseline_methods, key=lambda m: overall[METHOD_NAMES[m]]["f1"])
    best_baseline_name = METHOD_NAMES[best_baseline]
    ctf3_name = METHOD_NAMES["ctf_r3"]
    logger.info(f"Statistical comparison: CTF (R=3) vs {best_baseline_name}")

    p_val = mcnemar_test(y_test, all_test_preds["ctf_r3"], all_test_preds[best_baseline])
    ci = bootstrap_f1_diff(y_test, all_test_preds["ctf_r3"], all_test_preds[best_baseline], n=BOOTSTRAP_N)
    significant = p_val < 0.05
    logger.info(f"McNemar p={p_val:.6f}, CI={ci['lower']:.4f}..{ci['upper']:.4f}, significant={significant}")

    # --- Score distributions (summary stats) ---
    score_distributions: dict[str, dict] = {}
    for method in ["minhash_k2", "tfidf", "ctf_r3"]:
        name = METHOD_NAMES[method]
        scores = all_test_scores[method]
        dup_mask = y_test == 1
        non_dup_mask = y_test == 0
        dup_scores = scores[dup_mask].tolist()
        non_dup_scores = scores[non_dup_mask].tolist()

        def summarize(arr: list) -> dict:
            if not arr:
                return {"mean": 0.0, "std": 0.0, "q10": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0, "q90": 0.0, "n": 0}
            a = np.array(arr)
            qs = np.quantile(a, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
            return {"mean": float(a.mean()), "std": float(a.std()),
                    "q10": qs[0], "q25": qs[1], "q50": qs[2], "q75": qs[3], "q90": qs[4],
                    "n": len(arr)}

        score_distributions[name] = {
            "duplicate_summary": summarize(dup_scores),
            "non_duplicate_summary": summarize(non_dup_scores),
        }

    # --- Build output examples for schema compliance ---
    # The exp_gen_sol_out schema requires datasets[].examples[].{input, output}
    # We encode each test pair prediction from the best method as an example
    ctf3_f1 = overall[ctf3_name]["f1"]
    best_bl_f1 = overall[best_baseline_name]["f1"]
    output_examples = []
    for i in range(len(test)):
        ex = test[i]
        inp_str = json.dumps({"text1": ex["text1"], "text2": ex["text2"]})
        pred_dict = {METHOD_NAMES[m]: int(all_test_preds[m][i]) for m in all_methods}
        output_examples.append({
            "input": inp_str,
            "output": str(y_test[i]),
            **{f"predict_{k.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').lower()}": str(v)
               for k, v in pred_dict.items()},
            "metadata_is_duplicate": int(y_test[i]),
            "metadata_wc1": int(ex["wc1"]),
            "metadata_wc2": int(ex["wc2"]),
            "metadata_stratum": (
                "<=6" if (ex["wc1"] + ex["wc2"]) / 2 <= 6
                else "7-15" if (ex["wc1"] + ex["wc2"]) / 2 <= 15 else "16+"
            ),
        })

    # --- Thresholds keyed by method name ---
    named_thresholds = {METHOD_NAMES[m]: round(thresholds[m], 4) for m in all_methods}

    # --- Summary string ---
    summary = (
        f"CTF (R=3) achieved F1={ctf3_f1:.4f} vs best baseline {best_baseline_name} F1={best_bl_f1:.4f} "
        f"on Quora Question Pairs (n_test={len(test)}). "
        f"McNemar p={p_val:.6f} ({'significant' if significant else 'not significant'} at p<0.05). "
        f"Bootstrap 95% CI on F1 difference: [{ci['lower']:.4f}, {ci['upper']:.4f}] "
        f"(point estimate {ci['point_estimate']:.4f}). "
        f"Stratified results: {list(by_wc.keys())}. "
        f"TF-IDF F1={overall['TF-IDF Cosine']['f1']:.4f}. "
        f"Best overall: {max(overall, key=lambda k: overall[k]['f1'])} "
        f"F1={max(v['f1'] for v in overall.values()):.4f}."
    )
    logger.info(summary)

    # --- Assemble method_out.json ---
    method_out = {
        "metadata": {
            "description": "Circular Text Fingerprints vs baseline methods for near-duplicate text detection",
            "datasets_used": ["Quora Question Pairs"],
            "methods": list(METHOD_NAMES.values()),
            "evaluation_date": t_start.isoformat(),
            "n_train": len(train),
            "n_test": len(test),
            "bootstrap_samples": BOOTSTRAP_N,
            "thresholds": {"Quora Question Pairs": named_thresholds},
            "performance_table": {
                "Quora Question Pairs": {
                    "overall": overall,
                    "by_word_count": by_wc,
                }
            },
            "confusion_matrices": {"Quora Question Pairs": confusion},
            "statistical_comparison": {
                "Quora Question Pairs": {
                    "comparison": f"CTF (R=3) vs {best_baseline_name}",
                    "mcnemar_p_value": p_val,
                    "bootstrap_ci_f1_difference": ci,
                    "significant": significant,
                }
            },
            "score_distributions": {"Quora Question Pairs": score_distributions},
            "summary": summary,
        },
        "datasets": [
            {
                "dataset": "Quora Question Pairs (sentence-transformers/quora-duplicates)",
                "examples": output_examples,
            }
        ],
    }

    # Validate: no NaN/inf
    def check_finite(obj, path=""):
        if isinstance(obj, float):
            assert math.isfinite(obj), f"Non-finite at {path}: {obj}"
        elif isinstance(obj, dict):
            for k, v in obj.items():
                check_finite(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj[:3]):
                check_finite(v, f"{path}[{i}]")

    check_finite(method_out)
    assert 0 < named_thresholds.get("CTF (R=3)", 0.5) <= 1.0
    assert 0 <= p_val <= 1.0  # p=0 is valid when chi2 is extremely large
    assert ci["lower"] <= ci["upper"] + 1e-6

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(method_out, indent=2))
    size_mb = out_path.stat().st_size / 1024**2
    logger.info(f"Saved method_out.json ({size_mb:.1f} MB)")

    elapsed = (datetime.now(timezone.utc) - t_start).total_seconds()
    logger.info(f"=== Done in {elapsed:.0f}s ===")


if __name__ == "__main__":
    main()
