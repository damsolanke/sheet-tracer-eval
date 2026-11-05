# Plan: trace long seed rows as overlapping 7-point pieces

Written and saved 2026-11-05 16:28 CST, before any run of the wrapper on a released test seed or a fresh seed.

## 1. Why

scrollprize/sheet_tracer_7pt_gap60 was trained on 7-point fronts. On the held-out slab at threshold 0.4 the release measured coverage 0.54 / 0.64 at 7 / 9 points and 0.40 at 15. pieces/pieces.py (piece_trace) traces a seed row of more than 7 points as 7-point pieces starting every S points (the last piece ends at the last point), each with villa's own trace and the stock options, and merges them column by column: each dense column comes, at each step, from the piece still tracing that holds it closest to the piece's centre; a column no piece covers any more keeps its last position with confidence 0, as the benchmark pads a stopped trace. connected_keep, the model, its weights, the step and the threshold are unchanged; the benchmark scores the merged points. Seeds of 7 points or fewer pass through unchanged. Unit tests: pieces/test_pieces.py (5 pass).

## 2. Exploration and the one setting (done before this plan)

On 40 tuning windows at 15 points (pieces/tune_f840.json; tuning_seeds.py, generators [g, 7] from g = 1, every GT window disjoint from every released test window), threshold 0.4, paired against stock 0.4 on the same windows: stride 2 +0.133 [+0.072, +0.198] coverage, precision -0.002; stride 4 +0.108 [+0.048, +0.177], +0.002; stride 6 +0.116 [+0.055, +0.184], -0.002 (stock 0.4: 0.491 / 0.869, 33 real switches; every stride 37). Selection rule, set before the stride 6 result was read: the largest coverage gain among strides with precision change -0.010 or better; within 0.010 of it, the largest stride. **Setting: piece 7, stride 2** (5 pieces at 15 points, 2 at 9). At threshold 0.3 (stride 4, reported only): +0.155 over stock 0.3, precision -0.026. Best of three on 40 windows, so a smaller gain is expected.

## 3. Seeds and runs (pieces/series_test_*.txt; MPS fp32, nice 15, one GPU job at a time)

- Released 40 test seeds (release draw) at 15 points (decision), 9 points (scope) and 7 points (identity); references are the released runs A0_C0_f*_t0.4 and A0_C0_f840_t0.3.
- Fresh windows at 15 points: pieces/fresh_f840.json, 45 windows on 27 patches (slwh/fresh_seeds.py, generators [g, 7] from g = 201), every GT window disjoint from every released test window, every tuning window (7, 9 and 15 points), every earlier fresh window and each other; stock 0.4 is run on them the same day.
- Runs: pieces at 0.4 on the released seeds at 7, 9 and 15 points; stock 0.4 and pieces at 0.4 on the fresh windows; reported only: pieces at 0.3 on the released 15-point seeds against the released stock 0.3.

## 4. Decision rule

The setting passes when, at 15 points, against stock 0.4 on the same seeds (pooled = released and fresh pairs together):

1. pooled coverage gain is at least +0.05 with its 95 % paired bootstrap interval (2,000 resamples, generator seed 0) above zero, and the point estimate is above zero on the released seeds and on the fresh windows separately;
2. pooled precision change is -0.010 or better (point estimate);
3. pooled real switches are at most stock 0.4's plus 10 % of the pairs;
4. integrity: seed lists equal to the released and fresh lists; a seed whose run in either arm adds a noncached or unlisted chunk read is dropped from its pair (the per-trace counts are a lower bound: a chunk counted by an earlier trace is not counted again); if more than a quarter of the fresh pairs drop, the fresh part is reported as inconclusive and rules 1 to 3 apply to the released seeds alone;
5. identity: at 7 points the wrapper's per_trace.json equals the released A0_C0_f360_t0.4 byte for byte.

Scope at 9 points (released seeds; reported, decides scope only): the coverage interval against stock 0.4 does not lie entirely below zero, precision change -0.010 or better, real switches at most stock 0.4's plus 4.

Pass: the November entry is this wrapper (an opt-in option for villa's trace) with this evidence; the form and any villa pull request need the user's OK. Fail: report it plainly with every number; the fallback is the corrected evaluation text, which also needs the user's OK.

## 5. Data

Cached data only, nothing fetched, nothing re-hosted; nothing leaves this machine.
