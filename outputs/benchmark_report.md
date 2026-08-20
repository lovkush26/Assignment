# Benchmark: L1 (original) vs L2 (extended) pipeline

Measured on this machine with `python src/benchmark.py`, wall-clock time via `time.perf_counter()`,
single-threaded, cold Python process (no caching between runs).

## Dataset sizes

| | messages |
|---|---|
| L1 (original) | 900 |
| L2 batch | 180 |
| Combined (L2 run) | 1080 |

## Response time

| Run | Time (s) |
|---|---|
| L1 only - classify + extract + sensitive (3 parts, 900 msgs) | 0.2706 |
| L2 full - +priority +grouping +routing (6 parts, 1080 msgs) | 0.767 |
| Grouping only, fuzzy fallback **enabled** | 0.26 |
| Grouping only, fuzzy fallback **disabled** | 0.2256 |

L2 adds three new stages (priority scoring, grouping with a TF-IDF fallback, privacy routing) over
20% more messages, and still completes in a fraction of a second - no incremental/caching
architecture was needed at this dataset scale (documented as a limitation, not implemented).

## "Optimized component": TF-IDF fuzzy-match index (grouping.py / retrieval.py)

| | |
|---|---|
| Documents indexed | 51 (one per related-message group title) |
| Vocabulary terms | 187 |
| Sparse matrix size | 2788 bytes |

## Result-quality delta from adding the fuzzy fallback

| | count |
|---|---|
| Unresolved references **without** fuzzy fallback (exact subject match only) | 27 |
| Unresolved references **with** fuzzy fallback (as shipped) | 15 |
| Messages correctly recovered by the fuzzy layer | 12 |

Concretely, this is what resolves references like "our earlier model-results review" to the task
titled "Review the model results", or "the assignment" to "Upload the assignment" - phrasings that
share no exact normalized subject string with the original task title.

## Output volume (L2 full run, 1080 messages)

| Output | records |
|---|---|
| classifications | 1080 |
| tasks_events | 430 |
| sensitive_findings | 120 |
| priority decisions | 545 |
| related_message_groups | 51 |
| privacy_routing decisions | 1080 |
