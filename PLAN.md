# Plan: held-out, cross-protocol evaluation of `scrollprize/sheet_tracer_7pt_gap60`

> Published copy (Oct 28, 2026): references to a private lab notebook, a local folder and other jobs on the same machine were removed. The rules, thresholds and dates are unchanged.

Written 2026-10-21 at 10:26 CDT, before any benchmark result. The only tracer run so far is the pipeline smoke test (one seed, three steps), which says nothing about accuracy.

## 1. What is evaluated

| Item | Value |
|---|---|
| Checkpoint | `scrollprize/sheet_tracer_7pt_gap60`, revision `443497f0`, `last.pt`, sha256 `b1d133bc...53ec` |
| Run inside it | `remote_l0_neighbor`, iteration 24,682, 7-point fronts, 60 vx between points, 20 vx per step |
| Weights | EMA, as villa's `load_checkpoint` picks them; fp32 (villa enables bf16 autocast only on CUDA) |
| Code | villa `fe64ea3` (main on 2026-10-21); the sheet tracer's numerics are the same as at the #2045 merge `59a2081` |
| Benchmark | villa's `evaluation/benchmark.py`, called unchanged through `tracereval/run_bench.py` |

The team's held-out table (OPERATIONS.md, state of Oct 7) belongs to a different run, `remote_l0_points_ft`. No number for the public checkpoint on PHerc0139 or PHerc1667 has been published. "Reproduce" below therefore means landing in the same regime as the team's row, with tolerances sized for that (section 5), not matching it digit for digit.

## 2. Metrics

All from `benchmark.py` (`score`, `summarize`), per step over points with GT supervision, then averaged over the steps:

| Metric | Meaning |
|---|---|
| coverage | kept and on the seed's sheet, over scored points (a point is on the sheet within 6 vx, `loss.sheet_tolerance`) |
| precision | on the seed's sheet among kept points |
| traces with a real switch | traces with a row holding 3 or more kept off-sheet points, or one kept point more than 9 vx off |
| also reported | same_sheet, kept, caught, dist_p50, advance_p50, traces lost, traces stopped |

`kept` means confident and 4-connected to the seed row (`connected_keep`). These are the three numbers in the team's table (coverage / precision / traces with a real switch). Intervals: 95 % bootstrap over seeds (2,000 resamples) from the per-trace counts that `run_bench.py` records (`per_trace.json`); changes between conditions use paired bootstraps over the same seeds.

## 3. Benchmark settings

| Setting | Value | Why |
|---|---|---|
| Steps | 60 | the team's table |
| Threshold | 0.4 for the gate; 0.3 and 0.5 reported | 0.4 is the table's; villa's code comment calls 0.3 best for this model; 0.5 is the checkpoint's default |
| Front | 240, 480, 840 vx (5, 9, 15 points) for the gate; 360 vx (7 points) reported | the table's columns; 7 points is what this model was trained on |
| Seeds | 40, `--max-per-patch 0`, `--min-valid 0.95`, benchmark seed 0 | the team counts 48 switching traces on 38 PHerc0139 segments, so it allowed more than one seed per patch |
| Seed draws | `draw_seeds` inside `benchmark.main` (generator `[0, 7]`), deterministic | same code path as the team |
| Volume | read at level 0 of the prepared pyramid, as the checkpoint does | |
| Device | MPS if the check in section 5 passes, else CPU with 4 threads; nice 15 | the Mac is shared |

## 4. Data and conditions

PHerc0139 is the gate scroll because both of its scans that matter are partly cached: the 2.399 um training-protocol scan (20260102150214, level 2, about 9.6 um voxels) and the 9.362 um eligible-protocol scan (20250728140407, Paganin 1000, same as the 11 eligible 9.362 um volumes).

GT patches come from villa's `scripts/prepare_scroll.py` rules (every segment registered on the 2.399 um volume, coordinates times 0.25, nodes in empty CT plus 32 vx masked, held-out slab = middle 20 % of the segments' z extent, 256 vx gap to training nodes). `tracereval/prepare.py` applies the same rules without downloading: chunks that are not cached count as empty, so patches cover cached CT only.

| Condition | CT the model reads | Model voxel | Purpose |
|---|---|---|---|
| C0 (gate) | 2.399 um scan, level 2 | 9.6 um | training protocol; compared with the team's row |
| C1 | 9.362 um scan, native | 9.362 um | eligible protocol as users run it |
| C3 | 9.362 um scan resampled to the training voxel (s = 1 / 1.0283) | 9.63 um | eligible protocol at the training voxel size |
| C2 | 9.362 um scan resampled to s = 9.362 / 8.64 | 8.64 um | the geometry of the 11 eligible 8.64 um volumes read natively |

C1 to C3 reuse C0's seeds (`--seeds-from`, same patch, orientation and anchor). Patches go into each frame through the fitted affine from the October work (`windeval/results/transform_pherc0139_2399l2_to_9362.json`, 0.22 vx median residual, fitted on z 3900 to 4300 at level 2); C2 and C3 sample the 9.362 um scan once per crop sample (`tracereval/resample.py`), so no second interpolation enters. Held-out scoring in C1 to C3 maps scored points back to the 2.399 um frame, so every condition scores the same validation cells.

## 5. GO / NO-GO rule for the gate (C0)

GO only if all three hold:

1. **Integrity.** Every trace of every gated run reads only chunks that are cached or absent from the bucket: zero `noncached` and zero `unlisted` reads in `run_meta.json`. Each gated run has at least 40 seeds.
2. **Regime.** At threshold 0.4 and at each gated front size: coverage at least the team's value minus 0.12, and precision at least the team's value minus 0.04. For PHerc0139 (5 / 9 / 15 points): coverage at least 0.337 / 0.485 / 0.428, precision at least 0.940 / 0.940 / 0.917.
3. **Numerics** (only if MPS is used). On the first 2 seeds at 7 points and 20 steps, MPS and CPU fp32 keep the same points (at least 99 % identical keep flags) and their coverage differs by at most 0.01.

Coverage above the team's value plus 0.12 does not fail the gate. It is reported as a difference between the two models.

**Why these tolerances.** Coverage is close to a mean over traces, and a trace either holds the sheet or loses most of its remaining points once it switches or stops, so per-trace coverage spreads widely (standard deviation about 0.3, an estimate). That gives a standard error near 0.05 for 40 seeds and near 0.03 for the team's seeds (count not stated, likely 100 as in `bench_local.sh`). Their difference has a standard error near 0.06, so 0.12 is about two standard errors of pure seed noise. Within the team's own table, coverage on PHerc0139 moves by 0.15 between front sizes of a single model. Precision pools thousands of kept points per run; switching traces cluster its errors, so its seed noise is near 0.01, and 0.04 leaves room for a small model difference. The rule is one-sided because harness faults (wrong voxel units, misregistered patches, empty chunks read as zeros, broken normalization) all push traces off the sheet, lowering coverage and precision; none of them raises both. fp32 against the team's bf16 flips a few threshold decisions per trace (OPERATIONS.md), which the two margins cover.

NO-GO otherwise, naming the clause that failed.

### If the PHerc0139 slab is not cached (fixed before the feasibility numbers)

The cached PHerc0139 CT is the October winding-model mirror: level-2 z 3712 to 4480 around the meshed windings. The team's default held-out slab is z 4402 to 6234, and one 60-step trace advances 1,200 vx. Before any tracing, `tracereval/feasibility.py` draws the benchmark's own seeds without reading CT and checks which chunks each trace would read (every step's front box widened by the crop reach plus 32 vx of drift). The rule above is then applied unchanged to the first of these with at least 40 fully cached seeds at one or more of the table's front sizes, gating only those sizes:

- **A0.** PHerc0139, the team's held-out slab, `--all-val`.
- **A1.** PHerc0139, the cached band, `--held-out-scoring` (the team's other held-out definition: seed rows in validation cells of the 1024^3 split, only points over validation cells scored), against the PHerc0139 row.
- **A2.** PHerc1667, its default held-out slab where cached (z about 4864 to 5503), `--all-val`, against the team's PHerc1667 row (coverage at least 0.441 / 0.460 / 0.459, precision at least 0.940 / 0.932 / 0.917 for 5 / 9 / 15 points).

A GO from A0 or A1 opens C1 to C3 on PHerc0139, on seeds whose 9.362 um reads are cached too. A GO from A2 only shows that the harness reproduces the team's regime: C1 to C3 still need the data in section 7, and the one cached-data follow-up is PHerc1667's training scan read at 8.64 um against its native read on the same seeds (s = 9.596 / 8.64). If none of A0 to A2 has 40 cached seeds, the decision is NO-GO (data) and work stops at the download list.

## 6. After a GO: comparisons and the user guidance they decide

Same seeds and settings in every condition; per-seed paired differences with 95 % paired bootstrap intervals.

| Comparison | Question |
|---|---|
| C3 against C0 | what the eligible protocol costs at equal voxel size |
| C1 against C3 | whether a 9.362 um volume should be resampled (2.8 % voxel mismatch) |
| C2 against C3 | whether an 8.64 um volume should be resampled (11 % mismatch) |

Pre-registered reading: resampling is recommended for a voxel size if its coverage change against C3 has an interval below zero, or its precision drops by 0.01 or more with an interval below zero; otherwise the guidance is "no measurable effect at this sample size".

## 7. Data a later step needs

Added at 10:52 CDT after the feasibility runs; sections 1 to 6 are unchanged. None of A0, A1 or A2 has 40 cached seeds, so the rule gives NO-GO (data). The exact chunk keys, all present in the bucket and missing locally, were listed; nothing has been fetched.

| Option | Enables | Chunks | Size |
|---|---|---|---|
| `a1_first40` | A1 (PHerc0139 validation cells, 5-point fronts) and C0 to C3 on the same 40 seeds | 140 of the 2.399 um scan (level 2) + 153 of the 9.362 um scan | 0.57 GiB |
| `a0_slab` | A0, the team's held-out slab at all front sizes, C0 to C3, and an affine refit in the slab | 7,300 + 7,727 | 29.35 GiB |
| `pherc0343` | the real 8.64 um check on w047 / w048 | 1,506 | 2.94 GiB |

With `a1_first40` alone the gate can only test the 5-point column (coverage at least 0.337, precision at least 0.940), on seeds that all advance horizontally. `a0_slab` is the faithful version.

## 8. Compute

Heavy steps run only when no other large job holds the machine: nice 15, at most 4 CPU workers, MPS only with free memory and no fast swap growth. Smoke test cost: 0.5 s per step for a 7-point front on one CPU thread under a load average of about 60, 2.0 GiB peak memory. A 40-seed, 60-step run is about 2,400 steps per front size.

## 9. Amendment after the C0 gate

Written 2026-10-26 22:38 CDT, after the C0 gate gave NO-GO (22:14) and before any C1, C2 or C3 result exists (no run, no frame patch). Sections 1 to 8 stay as written, and the NO-GO of section 5 stands.

**Why.** The gate's comparator is a different run: the team's table belongs to `remote_l0_points_ft`, trained on 5 to 16-point fronts, while the public checkpoint was trained on 7-point fronts only. The 9-point column passes the gate. At every gated size, precision is within 0.011 of the team's, and kept on-sheet points sit a median 1.0 to 1.1 vx from the labels. Section 5 expects harness faults to lower precision as well as coverage, and precision did not drop; coverage at 5 and 15 points is lost to early stops.

**Scope.** C1, C2 and C3 as section 4 defines them (the October affine, checked inside the slab at 22:17: median residual 0.18 voxels), on C0's seeds (`--seeds-from` the C0 report at threshold 0.4 and the same front; the seeds are identical at every threshold), at 7 and 9 points only, thresholds 0.4 and 0.3: twelve runs. Settings and integrity rule as for C0: zero noncached and zero unlisted reads; a run that reads a noncached chunk is rerun after exactly those chunks are fetched, within the 1.5 GiB cap (0.963 GiB used so far). Before these runs, clause 3's numerics check is repeated on the C2 path (2 seeds, 7 points, 20 steps, MPS against CPU fp32, same thresholds 0.99 and 0.01); if it fails, C1 to C3 run on CPU with 4 threads.

**Comparisons and reading.** Section 6 unchanged: C3 against C0, C1 against C3 and C2 against C3, paired over the seeds both runs kept, with 95 % paired bootstraps (2,000 resamples). Each condition is also shown against C0. The reading is applied per cell (front size and threshold); all four cells are reported, none is primary, and where cells disagree the write-up says so. Resampling is recommended for a voxel size if its coverage change against C3 has an interval below zero, or its precision drops by 0.01 or more with an interval below zero; otherwise the guidance is "no measurable effect at this sample size".

**Label.** Every C1 to C3 result is exploratory: run after a NO-GO under this amendment, and reported as such wherever it appears.
