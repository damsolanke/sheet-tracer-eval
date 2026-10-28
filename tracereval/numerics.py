"""PLAN.md section 5, clause 3: MPS against CPU fp32 on the same seeds.

Two benchmark folders from ``tracereval.run_bench`` with the same arguments except ``--device``. Keep flags come from
the written traces (``tifxyz/seedNN_<model>``: villa writes every point outside ``connected_keep`` as -1); a trace
that stopped earlier on one device counts its missing rows as not kept. Coverage is the report's ``coverage_mean``.

  python -m tracereval.numerics CPU_RUN MPS_RUN --out numerics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

MIN_IDENTICAL = .99
MAX_COVERAGE_DIFF = .01


def keep_flags(folder: Path) -> np.ndarray:
    xyz = np.stack([tifffile.imread(folder/f'{c}.tif') for c in 'xyz'], -1)
    return np.all(xyz != -1, -1)


def compare(cpu: Path, mps: Path) -> dict:
    reports = [json.loads((run/'report.json').read_text()) for run in (cpu, mps)]
    if reports[0]['seeds'] != reports[1]['seeds']:
        raise ValueError('the two runs traced different seeds')
    name = next(iter(reports[0]['models']))
    seeds = []
    same = total = 0
    for i in range(len(reports[0]['seeds'])):
        a, b = (keep_flags(run/'tifxyz'/f'seed{i:02d}_{name}') for run in (cpu, mps))
        rows, cols = max(a.shape[0], b.shape[0]), a.shape[1]
        if b.shape[1] != cols:
            raise ValueError(f'seed {i}: column counts differ ({a.shape} vs {b.shape})')
        pa, pb = np.zeros((rows, cols), bool), np.zeros((rows, cols), bool)
        pa[:a.shape[0]], pb[:b.shape[0]] = a, b
        seeds.append(dict(seed=i, shape_cpu=list(a.shape), shape_mps=list(b.shape), flags=int(pa.size),
                          identical=int((pa == pb).sum()), kept_cpu=int(pa.sum()), kept_mps=int(pb.sum())))
        same += int((pa == pb).sum())
        total += int(pa.size)
    cov = [r['models'][name]['summary']['coverage_mean'] for r in reports]
    prec = [r['models'][name]['summary']['precision_mean'] for r in reports]
    identical = same/total
    out = dict(cpu=str(cpu), mps=str(mps), seeds=seeds, identical_fraction=identical, coverage_cpu=cov[0],
               coverage_mps=cov[1], coverage_diff=abs(cov[1]-cov[0]), precision_cpu=prec[0], precision_mps=prec[1],
               rule=dict(min_identical=MIN_IDENTICAL, max_coverage_diff=MAX_COVERAGE_DIFF))
    out['pass'] = bool(identical >= MIN_IDENTICAL and out['coverage_diff'] <= MAX_COVERAGE_DIFF)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cpu', type=Path)
    ap.add_argument('mps', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    result = compare(args.cpu, args.mps)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != 'seeds'}), flush=True)


if __name__ == '__main__':
    main()
