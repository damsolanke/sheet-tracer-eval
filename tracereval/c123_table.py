"""PLAN.md section 9 (amendment, exploratory): C1 to C3 per cell (front, threshold), paired as section 6 says.

Per run: ``c0_table.row`` (pooled metrics checked against the report, seed-bootstrap intervals, traced-row coverage,
stops) plus the number of C0 seeds the run kept. Paired (``analyze.paired``, 2,000 resamples over the seeds both
runs kept): C3 against C0, C1 against C3, C2 against C3 (section 6), and C1, C2 against C0 for reference.

Reading (section 6, unchanged), for C1 and C2 against C3 in each cell: resampling is recommended if the coverage
change has an interval below zero, or the precision change is -0.01 or less with an interval below zero; otherwise
"no measurable effect at this sample size".

  python -m tracereval.c123_table --runs-dir RUNS --out A0_C123_table.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tracereval import analyze
from tracereval.c0_table import row

FRONTS = (360, 480)
THRESHOLDS = (.4, .3)
PAIRS = (('C3', 'C0'), ('C1', 'C3'), ('C2', 'C3'), ('C1', 'C0'), ('C2', 'C0'))  # (run, reference)
VOXEL = {'C0': '9.6 um (2.399 um scan, level 2)', 'C1': '9.362 um native', 'C2': '8.64 um geometry',
         'C3': '9.362 um resampled to 9.63 um'}


def reading(p: dict) -> str:
    cov_hi = p['ci']['coverage'][1]
    prec, prec_hi = p['change']['precision'], p['ci']['precision'][1]
    if cov_hi < 0 or (prec <= -.01 and prec_hi < 0):
        return 'resample'
    return 'no measurable effect at this sample size'


def name(cond: str, front: int, threshold: float) -> str:
    return f'A0_{cond}_f{front}_t{threshold:g}'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    runs, pairs = [], []
    for front in FRONTS:
        for t in THRESHOLDS:
            for cond in ('C0', 'C3', 'C1', 'C2'):
                r = row(args.runs_dir/name(cond, front, t))
                runs.append(dict(r, cond=cond, threshold=t, voxel=VOXEL[cond]))
            for cond, ref in PAIRS:
                p = analyze.paired(args.runs_dir/name(ref, front, t), args.runs_dir/name(cond, front, t))
                entry = dict(p, front=front, points=int(round(front/60))+1, threshold=t, cond=cond, ref_cond=ref)
                if ref == 'C3' and cond in ('C1', 'C2'):
                    entry['reading'] = reading(p)
                pairs.append(entry)
    args.out.write_text(json.dumps(dict(label='exploratory: run after the C0 NO-GO under PLAN.md section 9',
                                        runs=runs, paired=pairs), indent=1))
    print('cond pts thr | seeds | coverage [CI] | precision [CI] | traced-row cov | real sw | stopped | noncached unlisted')
    for x in runs:
        print(f"{x['cond']} {x['points']:>2} {x['threshold']:.1f} | {x['seeds']} | {x['coverage']:.3f} "
              f"[{x['ci']['coverage'][0]:.3f}, {x['ci']['coverage'][1]:.3f}] | {x['precision']:.4f} "
              f"[{x['ci']['precision'][0]:.4f}, {x['ci']['precision'][1]:.4f}] | {x['coverage_traced']:.3f} | "
              f"{x['real_switches']} | {x['stopped']} | {x['noncached_reads']} {x['unlisted_reads']}")
    print('pts thr | run vs ref | pairs | coverage change [CI] | precision change [CI] | real sw run/ref | reading')
    for p in pairs:
        print(f"{p['points']:>2} {p['threshold']:.1f} | {p['cond']} vs {p['ref_cond']} | {p['pairs']} | "
              f"{p['change']['coverage']:+.3f} [{p['ci']['coverage'][0]:+.3f}, {p['ci']['coverage'][1]:+.3f}] | "
              f"{p['change']['precision']:+.4f} [{p['ci']['precision'][0]:+.4f}, {p['ci']['precision'][1]:+.4f}] | "
              f"{p['run_metrics']['real_switches']}/{p['ref_metrics']['real_switches']} | {p.get('reading', '')}")


if __name__ == '__main__':
    main()
