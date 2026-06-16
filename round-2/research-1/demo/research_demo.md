# Baselines and Statistical Testing for Near-Duplicate Detection: Complete Specifications

## Summary

This artifact documents three non-degenerate baseline algorithms (character 4-gram Jaccard, word-unigram Jaccard, TF-IDF cosine similarity) and rigorous statistical testing methods (McNemar's test, bootstrap confidence intervals) for evaluating near-duplicate detection on short text (8–20 words). The research shows that the original experiment's MinHash baseline was degenerate (threshold=0.00, classifying all pairs as duplicates), the short-text advantage claim was based on n=9 uninformative samples, and proper baselines require careful threshold tuning on validation data and statistical significance testing. The document provides implementation specifications, threshold tuning protocols, edge case handling, and documented pitfalls to enable valid experimental comparisons. Key findings: [1] Character 4-gram Jaccard is the standard shingling approach in IR with O(length × k) complexity and proven probabilistic approximation via MinHash [2] Word-unigram Jaccard is simpler and more interpretable but less sensitive to substring-level variations [3] TF-IDF cosine similarity captures semantic relatedness better than Jaccard but requires handling zero-vector edge cases [4] Threshold tuning via grid search on validation data is essential; using fixed thresholds (0.5 or 0.0) introduces confounds [5] McNemar's test is preferred for expensive-to-train classifiers on single test sets, especially when (b+c)≥25 for chi-squared approximation [6] Bootstrap confidence intervals (10,000 iterations, 2.5th/97.5th percentiles) provide distribution-free significance estimates for any metric [7] The Quora Question Pairs dataset has ~400K pairs (63% negative, 37% positive) with 10–25 word average length, requiring class imbalance awareness [8] Edge cases for short text (< k characters for n-grams, zero vectors for TF-IDF) must be handled with documented fallbacks [9] Prior degenerate baseline and insufficient statistical testing invalidate the original +0.0888 F1 improvement claim.

## Research Findings

## Baseline Algorithms

Three non-degenerate baseline algorithms are specified for evaluating near-duplicate detection on short text:

### 1. Character 4-Gram Jaccard Similarity
The character 4-gram shingle approach extracts all contiguous 4-character substrings from text, builds sets of these shingles, and computes Jaccard similarity as the ratio of intersection to union [1, 2, 5]. This is the standard shingling technique described in the Stanford NLP IR book [1] and is widely used in plagiarism detection and web deduplication [2, 7].

**Implementation:** Convert text to lowercase, generate all substrings of length k=4 using a sliding window, store in a set to remove duplicates, then compute J(A,B) = |A∩B| / |A∪B| [1, 2, 7]. The computational complexity is O(length × k) for shingle extraction and O(count) space for storing shingles [1].

**Key hyperparameter:** k=4 is standard in information retrieval; k=3 and k=5 are reasonable alternatives depending on text length [7]. The algorithm has theoretical guarantees: when k hash functions are applied to estimate Jaccard via MinHash, the expected error decreases as 1/k [4, 9].

**Threshold tuning:** Thresholds must be selected via grid search over [0.0, 1.0] on a held-out validation set to maximize F1 [1, 2, 7]. For texts shorter than k characters (e.g., 3-character words), the method produces fewer shingles; the standard approach is to return Jaccard=0.0 if texts differ and 1.0 if identical [7].

### 2. Word-Unigram Jaccard Similarity
This simpler baseline tokenizes text into words, builds a set of unique words (unigrams), and computes Jaccard similarity using the same formula [2, 7]. It is more robust to punctuation and whitespace variations than character n-grams but loses substring-level detail [2, 7].

**Implementation:** Split text on whitespace, convert to lowercase, optionally strip punctuation, and store unique words in a set [7]. Computational complexity is O(word_count), significantly faster than character n-grams [2, 7].

**No hyperparameters:** The algorithm is purely combinatorial; there are no tunable parameters beyond optional stopword removal [7].

**Threshold range:** Also [0.0, 1.0], though optimal thresholds for this baseline are typically lower than character 4-grams due to sparsity of word-level features in short texts (8–20 words contain at most 10–20 unique unigrams) [2, 7]. The method handles all edge cases gracefully: a single-word question yields Jaccard=0 if paired with a different word and 1.0 if identical [7].

### 3. TF-IDF Cosine Similarity
This baseline applies TF-IDF weighting to words (TF(term) × log(N / DF(term))) to build sparse vectors in word space, then computes cosine similarity as the normalized dot product [3, 6, 8]. TF-IDF captures semantic relatedness better than Jaccard because it downweights common terms and emphasizes distinguishing words [3, 6, 8].

**Implementation:** Use scikit-learn's TfidfVectorizer with default parameters (sublinear_tf=True for dampening, norm='l2' for L2 normalization), fit on training data, transform test documents, and compute pairwise cosine_similarity [3, 6, 8]. The cosine ranges [0, 1] after L2 normalization [3, 6].

**Hyperparameters:** TfidfVectorizer defaults are good starting points: sublinear_tf=True (dampens term frequency), use_idf=True, norm='l2' (ensures normalized vectors) [3, 6, 8].

**Edge cases:** Single-word texts or stop-word-only texts produce sparse or zero vectors; cosine_similarity returns NaN for zero vectors. Standard fallback: replace NaN with 0.0 [3, 6, 8].

## Threshold Tuning Protocol

All three baselines require threshold tuning to convert similarity scores to binary predictions (duplicate/non-duplicate). The standard protocol is grid search [11, 12]:

1. **Sweep thresholds:** Iterate τ ∈ [0.0, 1.0] in steps of 0.01 or 0.001 on the validation set (not test set, to avoid data leakage) [11, 12].
2. **Compute metrics:** For each τ, classify pairs as positive if similarity ≥ τ, compute precision, recall, and F1 [11, 12].
3. **Select optimal:** Choose τ that maximizes F1 on validation data [11, 12].
4. **Evaluate on test:** Apply the selected τ to the held-out test set to report final performance [11, 12].

**Critical pitfall:** The MinHash baseline in the original experiment used τ=0.00, meaning all pairs were classified as duplicates. This yields precision = 37% (proportion of actual duplicates), recall = 100%, and F1 ≈ 0.54 — a degenerate baseline that should not be used [original experiment analysis].

**Handling class imbalance:** On the Quora Question Pairs dataset (~63% negative, 37% positive), the default 0.5 threshold biases toward the majority class. Threshold-moving is necessary to find the optimal F1 for each baseline independently [11, 12].

## Statistical Testing Methodology

### McNemar's Test
McNemar's test is a non-parametric test for comparing two classifiers on the same test set [10, 13, 14]. It is based on a 2×2 contingency table:

|  | Classifier B Correct | Classifier B Wrong |
|---|---|---|
| **Classifier A Correct** | n00 | n01 (b) |
| **Classifier A Wrong** | n10 (c) | n11 |

The test statistic is χ² = (|b - c| - 1)² / (b + c) under the chi-squared approximation [10, 13, 14]. The null hypothesis is that both classifiers have identical error distributions (b = c) [10, 13, 14].

**When to use exact vs. approximate:** If (b + c) < 25, use the exact binomial test (exact=True); if (b + c) ≥ 25, use the chi-squared approximation (exact=False) [10, 13, 14]. This is the recommendation from Dietterich (1998), who evaluated five statistical tests for classifier comparison and found McNemar's test to be the most powerful and recommended for expensive-to-train models on single test splits [Dietterich PDF reference].

**Implementation:** `statsmodels.stats.contingency_tables.mcnemar(table, exact=True/False, correction=True)` [10].

**Interpretation:** p-value < 0.05 indicates a statistically significant difference. Report as: "Classifier A F1 = 0.75, Classifier B F1 = 0.70, McNemar p-value = 0.032." [10, 13, 14]

### Bootstrap Confidence Intervals
Bootstrap provides distribution-free confidence intervals for any metric, including F1 [17, 18, 19]. The standard procedure:

1. **Resample:** Resample the test set WITH REPLACEMENT n_iterations times (default: 10,000) [17, 18, 19].
2. **Compute:** Calculate F1 (or other metric) on each resample [17, 18, 19].
3. **Percentiles:** Extract the 2.5th and 97.5th percentiles of the distribution to form a 95% CI [17, 18, 19].

**Advantage:** Bootstrap makes no parametric assumptions and works for any metric. Non-overlapping CIs across two models indicate approximately p < 0.05 significance [17, 18, 19].

**Sample size implications:** Small test sets (n < 100) have wide CIs (±0.10–0.20); narrow CIs (±0.02) require n > 500 test pairs [17, 18, 19]. This is important for the original experiment's short-text stratum claim: with n=9 pairs, confidence intervals are approximately ±0.3, rendering any point estimate meaningless [statistical power analysis].

**Implementation:** Use `numpy.random.choice(n_test, size=n_test, replace=True)` to draw resample indices, compute the metric, and use `numpy.percentile(results, [2.5, 97.5])` [17, 18, 19].

**Cost:** Computationally cheaper than retraining: only metric recomputation, not model retraining [17, 18, 19].

### Reporting Results
Present both point estimates and uncertainty: "CTF F1 = 0.75 (95% CI: 0.72–0.78) vs. Baseline F1 = 0.70 (95% CI: 0.67–0.73)." [17, 18, 19] Non-overlapping CIs indicate p < 0.05. However, practical significance should be assessed separately: an F1 difference of 0.02 may be statistically significant but practically weak [17, 18, 19].

## Quora Question Pairs Benchmark

The Quora Question Pairs (QQP) dataset is a standard NLU benchmark [15, 16]:

- **Size:** ~400,000 question pairs, typically split ~364K train / 36K test [15, 16].
- **Class distribution:** ~63% negative (non-duplicate), 37% positive (duplicate) — imbalanced [15, 16].
- **Question length:** Average 10–25 words, aligning with the target range of 8–20 words [15, 16].
- **Baseline results:** TF-IDF baseline F1 ≈ 0.60–0.70; BERT fine-tuned F1 ≈ 0.85–0.90 [15, 16].

**Sampling strategy for this evaluation:** To eliminate class imbalance effects on threshold tuning, create a balanced subset: 5,000 positive pairs + 5,000 negative pairs (50%-50%). Split this as: 70% train (3,500), 15% validation (750), 15% test (750) with stratification to maintain class balance in each split [artifact plan].

## Edge Case Handling Checklist

1. **Character 4-gram short text:** For texts < 4 characters, produce fewer or zero shingles. Standardize: return Jaccard=0.0 if texts differ, 1.0 if identical [7].
2. **Word-unigram graceful handling:** Single-word texts produce single-word sets. Handles all edge cases naturally without special logic [7].
3. **TF-IDF zero vectors:** Replace NaN (from zero-vector cosine similarity) with 0.0 [3, 6, 8].
4. **Threshold boundaries:** At τ=0.0, all pairs predicted positive; at τ=1.0, all pairs predicted negative. Grid search must cover the full [0.0, 1.0] range [11, 12].

## Known Pitfalls and Corrections

**Degenerate MinHash baseline:** The original experiment used τ=0.00, classifying all pairs as duplicates. This yields F1 ≈ 0.54 (37% precision, 100% recall), invalidating the +0.0888 F1 improvement claim. Proper baselines require threshold tuning [original experiment analysis].

**Short-text sample size:** The original claimed a "short-text advantage" based on n=9 test pairs in the 8–20 word stratum. This is statistically uninformative; F1 estimates have confidence intervals of ±0.2–0.3 at this scale, making any point difference meaningless [statistical power analysis].

**Missing statistical testing:** The prior experiment lacked McNemar's test or bootstrap CIs. Without significance testing, one cannot distinguish signal from noise, especially on small samples [13, 14, 17, 18].

**Class imbalance masking differences:** Without independent threshold tuning for each baseline, the default 0.5 threshold may favor one algorithm over another for spurious reasons unrelated to true performance [11, 12].

## Follow-Up Investigation Questions

1. **Once baselines are implemented and tuned on validation data, what are the observed F1 scores for each algorithm?** Do optimal thresholds differ significantly (e.g., character 4-gram τ=0.65 vs. word-unigram τ=0.30 vs. TF-IDF τ=0.75)?  

2. **How do baseline performances vary by text length stratum** (8–12 words vs. 13–17 words vs. 18–20 words)? Do shorter texts systematically favor one baseline?  

3. **When evaluated with McNemar's test and bootstrap CIs on n=750 test pairs, does CTF achieve statistically significant improvements** (non-overlapping CIs) over all three properly-tuned, non-degenerate baselines?

## Sources

[1] [Near-duplicates and shingling — Stanford NLP Group Information Retrieval Book](https://nlp.stanford.edu/IR-book/html/htmledition/near-duplicates-and-shingling-1.html) — Authoritative reference on shingling and Jaccard similarity for near-duplicate detection. Defines k-shingles, Jaccard coefficient, and MinHash probabilistic approximation with theoretical guarantees.

[2] [Finding near-duplicates with Jaccard similarity and MinHash](https://blog.nelhage.com/post/fuzzy-dedup/) — Comprehensive explanation of Jaccard similarity and MinHash approximation. Covers practical implementation details, sampling theory, and scale-efficient deduplication strategies used in GPT-3 dataset preparation.

[3] [TfidfVectorizer — scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) — API documentation and defaults for TF-IDF vectorization including sublinear_tf, norm='l2', and cosine similarity computation for text similarity.

[4] [MinHash — Wikipedia](https://en.wikipedia.org/wiki/MinHash) — Mathematical foundations of MinHash algorithm, min-wise independence, and probabilistic estimation of Jaccard similarity with k independent hash functions.

[5] [Shingling for Similarity and Plagiarism Detection — DZone](https://dzone.com/articles/shingling-for-similarity-and-plagiarism-detection) — Comprehensive tutorial on shingling technique, character n-grams, word n-grams, and implementation of Jaccard similarity with working code examples.

[6] [Cosine Similarity and TFIDF — Medium](https://medium.com/web-mining-is688-spring-2021/cosine-similarity-and-tfidf-c2a7079e13fa) — Practical explanation of TF-IDF weighting, cosine similarity metric, and L2 normalization for document similarity computation.

[7] [Character N-gram F-score: A Comprehensive Guide — Shade Coder](https://www.shadecoder.com/topics/character-n-gram-f-score-a-comprehensive-guide-for-2025) — Detailed guide on character n-gram extraction, shingle selection, and handling edge cases for short text similarity.

[8] [Python: tf-idf-cosine to find document similarity — Stack Overflow](https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity) — Practical code examples for TF-IDF vectorization and cosine similarity computation using scikit-learn.

[9] [Jaccard index — Wikipedia](https://en.wikipedia.org/wiki/Jaccard_index) — Mathematical definition and properties of Jaccard similarity, including set-based formulation and asymptotic bounds.

[10] [statsmodels.stats.contingency_tables.mcnemar — statsmodels documentation](https://www.statsmodels.org/dev/generated/statsmodels.stats.contingency_tables.mcnemar.html) — API documentation for McNemar's test implementation, including exact binomial vs. chi-squared options and usage examples.

[11] [3.3. Tuning the decision threshold for class prediction — scikit-learn documentation](https://scikit-learn.org/stable/modules/classification_threshold.html) — Guidelines for threshold selection in binary classification, precision-recall tradeoff, and F1 optimization for imbalanced data.

[12] [[Imbalanced Datasets] Threshold Tuning with ROC and PR Curves — Medium](https://medium.com/@eric.likp/imbalanced-datasets-threshold-tuning-with-roc-and-pr-curves-de4eb7417e97) — Practical guidance on threshold tuning for imbalanced classification using precision-recall curves and F1 optimization.

[13] [McNemar's test for classifier comparisons — mlxtend documentation](https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/) — Detailed explanation of McNemar's test, 2×2 contingency table, and interpretation of results for classifier comparison.

[14] [McNemar's Test to evaluate Machine Learning Classifiers with Python — Towards Data Science](https://towardsdatascience.com/mcnemars-test-to-evaluate-machine-learning-classifiers-with-python-9f26191e1a6b) — Practical tutorial on computing and interpreting McNemar's test with Python examples.

[15] [Question Pairs Dataset — Kaggle](https://www.kaggle.com/datasets/quora/question-pairs-dataset) — Quora Question Pairs dataset description, size (~400K pairs), and class distribution information.

[16] [First Quora Dataset Release: Question Pairs — Quora Data Blog](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs) — Original dataset announcement with dataset statistics, benchmark task description, and baseline results.

[17] [Confidence Intervals for Performance Metrics — tidymodels](https://tidymodels.org/learn/models/bootstrap-metrics/) — Bootstrap methodology for computing confidence intervals on machine learning metrics, including interpretation and sample size considerations.

[18] [How to calculate confidence intervals for performance metrics using Bootstrap — Towards Data Science](https://towardsdatascience.com/get-confidence-intervals-for-any-model-performance-metrics-in-machine-learning-f9e72a3becb2) — Practical guide to bootstrap confidence intervals with Python implementation for any metric (F1, precision, recall).

[19] [Confidence Intervals for Evaluation in Machine Learning — GitHub](https://github.com/luferrer/ConfidenceIntervals) — Reference implementation of bootstrap confidence intervals for machine learning metrics and evaluation.

## Follow-up Questions

- Once all three baselines are properly implemented and thresholds are tuned independently on the Quora Question Pairs validation set, what are the resulting F1 scores, optimal thresholds, and bootstrap 95% confidence intervals for each algorithm? Do the three baselines cluster into different performance tiers?
- How do baseline F1 scores vary across text length strata (8–12 words vs. 13–17 words vs. 18–20 words)? Does the short-text regime (8–12 words) systematically favor word-unigram over character 4-gram due to sparsity, or are differences negligible?
- When Circular Text Fingerprints are evaluated on the same test set and compared via McNemar's test and bootstrap CIs, do non-overlapping confidence intervals confirm statistically significant improvements over all three non-degenerate baselines, and are the improvements large enough to be practically meaningful (ΔF1 > 0.05)?

---
*Generated by AI Inventor Pipeline*
