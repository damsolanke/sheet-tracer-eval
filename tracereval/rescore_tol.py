"""Coverage and precision of a benchmark run rescored at another same-sheet tolerance (sensitivity check only).

The tolerance (``loss.sheet_tolerance``, 6) is in model voxels, so a condition whose model voxel is smaller judges
"on the sheet" more strictly in micrometres. This recomputes ``cover`` per trace and step from the written traces and
GT windows exactly as ``benchmark.rescore`` reads them (kept points are the written ones; ``landing`` against the GT
tifxyz), first at the run's own tolerance, which must reproduce ``per_trace.json`` exactly, then at ``--tolerance``.
``ok`` and ``kept`` do not depend on the tolerance. Writes ``per_trace_tol<T>.json`` next to the run for
``analyze``-style pooling.

  python -m tracereval.rescore_tol RUN --tolerance 6.685
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tracereval import env

env.setup(threads=2)

import torch  # noqa: E402

from vesuvius.neural_tracing.tracers.sheets.evaluation.benchmark import _read_xyz  # noqa: E402
from vesuvius.neural_tracing.tracers.sheets.losses import landing  # noqa: E402

BASE_TOLERANCE = 6.


def cover_counts(run: Path, seed: int, name: str, tolerance: float) -> np.ndarray:
    gt = _read_xyz(run/'tifxyz'/f'seed{seed:02d}_gt')
    rows = _read_xyz(run/'tifxyz'/f'seed{seed:02d}_{name}')[1:]
    kept = torch.from_numpy((rows != -1).all(-1))
    _, dist, same, ok = landing(torch.from_numpy(rows).reshape(1, -1, 3), torch.from_numpy(gt)[None],
                                torch.from_numpy((gt != -1).all(-1))[None], tolerance)
    same, ok = same[0].reshape(kept.shape), ok[0].reshape(kept.shape)
    return (ok & kept & same).sum(1).numpy().astype(int)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', type=Path)
    ap.add_argument('--tolerance', type=float, required=True)
    args = ap.parse_args(argv)
    report = json.loads((args.run/'report.json').read_text())
    if report.get('held_out_scoring'):
        raise SystemExit('held-out scoring runs are not supported')
    name = next(iter(report['models']))
    traces = json.loads((args.run/'per_trace.json').read_text())
    out, mismatched = [], 0
    for i, trace in enumerate(traces):
        base = cover_counts(args.run, i, name, BASE_TOLERANCE)
        if base.tolist() != trace['cover']:
            mismatched += 1
        new = cover_counts(args.run, i, name, args.tolerance)
        out.append(dict(trace, cover=new.tolist()))
    if mismatched:
        raise SystemExit(f'{mismatched} traces do not reproduce their cover counts at {BASE_TOLERANCE}: not used')
    path = args.run/f'per_trace_tol{args.tolerance:g}.json'
    path.write_text(json.dumps(out))
    print(f'{args.run.name}: {len(out)} traces reproduce at {BASE_TOLERANCE}; rescored at {args.tolerance:g} -> {path}')


if __name__ == '__main__':
    main()
