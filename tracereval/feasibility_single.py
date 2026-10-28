"""Cached-chunk feasibility of benchmark seeds in a single frame (no second scan), e.g. PHerc1667's held-out slab.

Same seed draws and per-step read estimate as ``tracereval.feasibility``; no CT is read or downloaded (bucket row
listings only, with ``--list-bucket``).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from tracereval import env

env.setup(threads=1)

from tracereval import chunks  # noqa: E402
from tracereval import feasibility as F  # noqa: E402
from vesuvius.neural_tracing.tracers.sheets.evaluation.benchmark import draw_seeds  # noqa: E402
from vesuvius.neural_tracing.tracers.sheets.sampling import view_radius  # noqa: E402


def classify(local: Path, remote: str, keys: set, list_bucket: bool) -> dict:
    out = dict(cached=0, absent=0, noncached=0, unlisted=0)
    rows = {}
    for key in keys:
        kind = chunks.classify(local, key)
        if kind == 'unlisted' and list_bucket:
            if key[0] not in rows:
                rows[key[0]] = F.list_row(remote, key[0])
            kind = 'noncached' if f'{key[1]}/{key[2]}' in rows[key[0]] else 'absent'
        out[kind] += 1
    out['download_gib'] = round((out['noncached']+out['unlisted'])*chunks.CHUNK_BYTES/2**30, 2)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--patches', type=Path, required=True)
    ap.add_argument('--local', type=Path, required=True, help='local level directory the model reads (level 0)')
    ap.add_argument('--remote', required=True, help='bucket path of that level, e.g. PHerc1667/volumes/X.zarr/2')
    ap.add_argument('--fronts', type=float, nargs='+', default=[240., 360., 480., 840.])
    ap.add_argument('--count', type=int, default=40)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--drifts', type=float, nargs='+', default=[0., 16., 32.])
    ap.add_argument('--held-out-scoring', action='store_true',
                    help='seed rows in validation cells (benchmark --held-out-scoring) instead of --all-val')
    ap.add_argument('--max-per-patch', type=int, default=0)
    ap.add_argument('--list-bucket', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = F.load_cfg()
    base = view_radius(cfg.model)+cfg.data.step+4
    report = dict(args={k: str(v) for k, v in vars(args).items()}, read_pad_without_drift=base,
                  villa_commit=env.VILLA_COMMIT, fronts={})
    for front in args.fronts:
        t0 = time.time()
        c = copy.deepcopy(cfg)
        c.data.patches_dir = [str(args.patches)]
        seeds, spec, data = draw_seeds(c, args.count, args.steps, front, .95, all_val=not args.held_out_scoring,
                                       max_per_patch=args.max_per_patch, seed_split=args.held_out_scoring)
        per_seed, union = [], set()
        for w in seeds:
            row = dict(meta=w['meta'], axis=w['meta']['axis'],
                       z=[round(float(w['reach_lo'][2])), round(float(w['reach_hi'][2]))])
            for drift in args.drifts:
                keys = F.step_keys(w, spec, data, args.steps, base+drift)
                cls = classify(args.local, args.remote, keys, args.list_bucket)
                row[f'missing_drift{int(drift)}'] = cls['noncached']+cls['unlisted']
                row[f'chunks_drift{int(drift)}'] = len(keys)
                if drift == max(args.drifts):
                    union |= keys
            per_seed.append(row)
        entry = dict(drawn=len(seeds), seconds=round(time.time()-t0, 1),
                     fully_cached={f'drift{int(d)}': sum(r[f'missing_drift{int(d)}'] == 0 for r in per_seed)
                                   for d in args.drifts},
                     union_at_max_drift=classify(args.local, args.remote, union, args.list_bucket), seeds=per_seed)
        report['fronts'][str(int(front))] = entry
        print(f"front {int(front)}: drawn {entry['drawn']}/{args.count}, fully cached {entry['fully_cached']}, "
              f"union {entry['union_at_max_drift']} ({entry['seconds']} s)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
