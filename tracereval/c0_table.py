"""The C0 runs (every front and threshold) as one table, plus paired threshold changes on the same seeds.

Per run: ``analyze.summary`` (pooled coverage and precision checked against the report, 95 % seed-bootstrap
intervals), coverage over traced rows only (``cover`` over ``seen``: the same pooling, but rows after an early
stop leave the denominator, which separates stopping from leaving the sheet), and the stop steps. Paired changes:
``analyze.paired`` of each threshold against 0.4 at the same front (identical seeds).

  python -m tracereval.c0_table --runs-dir RUNS --out c0_table.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tracereval import analyze

FRONTS = (240, 360, 480, 840)
THRESHOLDS = (.3, .4, .5)


def traced_coverage(traces: list, index=None) -> float:
    index = np.arange(len(traces)) if index is None else index
    seen = np.sum([traces[i]['seen'] for i in index], 0).astype(float)
    cover = np.sum([traces[i]['cover'] for i in index], 0).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        return float(np.nanmean(np.where(seen > 0, cover/np.maximum(seen, 1), np.nan)))


def traced_ci(traces: list, n_boot: int = 2000, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    draws = [traced_coverage(traces, rng.integers(0, len(traces), len(traces))) for _ in range(n_boot)]
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def row(run: Path) -> dict:
    s = analyze.summary(run)
    r = analyze.load(run)
    steps = [t['steps_traced'] for t in r['meta']['traces']]
    stops = [t['stopped'] for t in r['traces']]
    return dict(s, coverage_traced=traced_coverage(r['traces']), coverage_traced_ci=traced_ci(r['traces']),
                steps_traced_median=float(np.median(steps)), steps_traced_mean=float(np.mean(steps)),
                stopped_before_10=int(sum(x is not None and x <= 10 for x in stops)),
                full_length=int(sum(x is None for x in stops)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    rows, pairs = [], []
    for front in FRONTS:
        for t in THRESHOLDS:
            run = args.runs_dir/f'A0_C0_f{front}_t{t:g}'
            rows.append(dict(row(run), threshold=t))
        ref = args.runs_dir/f'A0_C0_f{front}_t0.4'
        for t in (.3, .5):
            p = analyze.paired(ref, args.runs_dir/f'A0_C0_f{front}_t{t:g}')
            pairs.append(dict(front=front, points=int(round(front/60))+1, threshold=t, against=.4, **p))
    args.out.write_text(json.dumps(dict(rows=rows, paired_thresholds=pairs), indent=1))
    print('points thr | coverage [CI] | precision [CI] | traced-row coverage [CI] | real sw | lost | stopped | '
          'full length | steps traced median | noncached unlisted')
    for x in rows:
        print(f"{x['points']:>2} {x['threshold']:.1f} | {x['coverage']:.3f} [{x['ci']['coverage'][0]:.3f}, "
              f"{x['ci']['coverage'][1]:.3f}] | {x['precision']:.4f} [{x['ci']['precision'][0]:.4f}, "
              f"{x['ci']['precision'][1]:.4f}] | {x['coverage_traced']:.3f} [{x['coverage_traced_ci'][0]:.3f}, "
              f"{x['coverage_traced_ci'][1]:.3f}] | {x['real_switches']} | {x['lost']} | {x['stopped']} | "
              f"{x['full_length']} | {x['steps_traced_median']:.0f} | {x['noncached_reads']} {x['unlisted_reads']}")
    print('paired threshold changes against 0.4 (same seeds): points thr | pairs | coverage change [CI] | '
          'precision change [CI]')
    for p in pairs:
        print(f"{p['points']:>2} {p['threshold']:.1f} | {p['pairs']} | {p['change']['coverage']:+.3f} "
              f"[{p['ci']['coverage'][0]:+.3f}, {p['ci']['coverage'][1]:+.3f}] | {p['change']['precision']:+.4f} "
              f"[{p['ci']['precision'][0]:+.4f}, {p['ci']['precision'][1]:+.4f}]")


if __name__ == '__main__':
    main()
