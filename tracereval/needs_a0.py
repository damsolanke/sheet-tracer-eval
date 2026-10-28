"""Chunk keys the A0 seeds may read that are in the bucket but not on disk, for a fetch list (no CT is read).

Same seeds and read estimate as ``feasibility_a0`` (``draw_seeds`` with generator ``[0, 7]``, ``--max-per-patch 0``,
``--min-valid 0.95``; every step's GT front box widened by the crop reach plus ``--drift``), in the 2.399 um level-2
frame (C0) and through the fitted affine in the 9.362 um scan (the C3 path, the widest of C1 to C3). Rows must have
``.windeval_present`` listings (``tracereval.listings``); a key in an unlisted row stops the script.

  python -m tracereval.needs_a0 --patches DIR --drifts 32 64 96 --out needs.json
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from tracereval import env

env.setup(threads=1)

from tracereval import chunks  # noqa: E402
from tracereval import feasibility as F  # noqa: E402
from vesuvius.neural_tracing.tracers.sheets.evaluation.benchmark import draw_seeds  # noqa: E402
from vesuvius.neural_tracing.tracers.sheets.sampling import view_radius  # noqa: E402


def missing(volume: str, keys: set) -> tuple[set, dict]:
    info = F.VOLUMES[volume]
    out, counts = set(), dict(cached=0, absent=0, noncached=0, unlisted=0)
    for key in keys:
        kind = chunks.classify(info['local'], key)
        counts[kind] += 1
        if kind in ('noncached', 'unlisted'):
            out.add(key)
    return out, counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--patches', type=Path, required=True)
    ap.add_argument('--fronts', type=float, nargs='+', default=[240., 360., 480., 840.])
    ap.add_argument('--count', type=int, default=40)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--drifts', type=float, nargs='+', default=[32.])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = F.load_cfg()
    base = view_radius(cfg.model)+cfg.data.step+4
    matrix = F.affine()
    scale9 = float(np.cbrt(abs(np.linalg.det(matrix[:3, :3]))))
    seeds_by_front = {}
    for front in args.fronts:
        c = copy.deepcopy(cfg)
        c.data.patches_dir = [str(args.patches)]
        seeds_by_front[front] = draw_seeds(c, args.count, args.steps, front, .95, all_val=True, max_per_patch=0,
                                           seed_split=False)
    report = dict(patches=str(args.patches), fronts=args.fronts, count=args.count, steps=args.steps, base_pad=base,
                  affine=str(F.AFFINE), villa_commit=env.VILLA_COMMIT, drifts={})
    for drift in args.drifts:
        pad = base+drift
        need = {'l2': set(), '9362': set()}
        per_front = {}
        for front, (seeds, spec, data) in seeds_by_front.items():
            k2, k9 = set(), set()
            seeds_missing = []
            for w in seeds:
                s2 = F.step_keys(w, spec, data, args.steps, pad)
                s9 = F.step_keys(w, spec, data, args.steps, pad, matrix, scale9)
                m2, c2 = missing('l2', s2)
                m9, c9 = missing('9362', s9)
                if c2['unlisted'] or c9['unlisted']:
                    raise SystemExit(f'unlisted rows at front {front}, drift {drift}: list them first')
                k2 |= m2
                k9 |= m9
                seeds_missing.append([len(m2), len(m9)])
            per_front[str(int(front))] = dict(seeds=len(seeds), l2_missing=len(k2), x9362_missing=len(k9),
                                              seeds_fully_cached_l2=sum(a == 0 for a, _ in seeds_missing),
                                              seeds_fully_cached_both=sum(a == 0 and b == 0 for a, b in seeds_missing))
            need['l2'] |= k2
            need['9362'] |= k9
        gib = (len(need['l2'])+len(need['9362']))*chunks.CHUNK_BYTES/2**30
        report['drifts'][str(int(drift))] = dict(
            pad=pad, per_front=per_front, l2_keys=sorted(need['l2']), x9362_keys=sorted(need['9362']),
            l2_count=len(need['l2']), x9362_count=len(need['9362']), total_gib=round(gib, 3))
        print(f'drift {drift:g}: L2 {len(need["l2"])} + 9.362 {len(need["9362"])} chunks = {gib:.3f} GiB; '
              f'per front {per_front}', flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else str(o)))
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
