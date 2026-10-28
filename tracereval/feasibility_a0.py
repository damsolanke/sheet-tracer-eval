"""PLAN.md section 5 feasibility for A0 (PHerc0139, the team's held-out slab, ``--all-val``) on the prepared patches.

``feasibility.seeds_report`` unchanged: the benchmark's own seed draws (``draw_seeds``, generator ``[0, 7]``,
``--max-per-patch 0``, ``--min-valid 0.95``) and each trace's reads estimated as every step's GT front box widened by
the crop reach plus ``--drift`` voxels, in the 2.399 um level-2 frame (C0) and through the fitted affine in the
9.362 um scan (the C3 path, the widest of C1 to C3). No CT is read; row listings come from the mirrors'
``.windeval_present`` files (``tracereval.listings``) or, for rows without one, from the bucket (keys only).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tracereval import env

env.setup(threads=1)

from tracereval import feasibility as F  # noqa: E402
from vesuvius.neural_tracing.tracers.sheets.sampling import view_radius  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--patches', type=Path, required=True, help='heldout_patches of prepare.py for the slab')
    ap.add_argument('--fronts', type=float, nargs='+', default=[240., 360., 480., 840.])
    ap.add_argument('--count', type=int, default=40)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--drift', type=float, default=32.)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = F.load_cfg()
    pad = view_radius(cfg.model)+cfg.data.step+4+args.drift
    report = dict(view_radius=view_radius(cfg.model), read_pad=pad, villa_commit=env.VILLA_COMMIT,
                  patches=str(args.patches), protocols={})
    for front in args.fronts:
        key = f'A0_slab_allval_front{int(front)}'
        r = F.seeds_report(cfg, args.patches, args.count, args.steps, front, True, False, 0, pad, True)
        report['protocols'][key] = r
        print(f"{key} ({r['points']} points): drawn {r['drawn']}/{args.count}, fully cached L2 "
              f"{r['seeds_fully_cached_l2']}, L2 and 9.362 {r['seeds_fully_cached_both']}; union L2 {r['union_l2']}, "
              f"union 9.362 {r['union_9362']} ({r['seconds']} s)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
