"""A1 seeds (PHerc0139 validation cells in the cached band): chunks their traces would read in both scans.

Re-draws the seeds of ``feasibility_single --held-out-scoring`` (same generator, same order) and estimates each
trace's reads in the 2.399 um level-2 frame and, through the fitted affine, in the 9.362 um scan (the C0 / C3 path,
the longest of the four conditions). No CT is read; bucket row listings only.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from tracereval import env

env.setup(threads=1)

from tracereval import feasibility as F  # noqa: E402
from vesuvius.neural_tracing.tracers.sheets.evaluation.benchmark import draw_seeds  # noqa: E402
from vesuvius.neural_tracing.tracers.sheets.sampling import view_radius  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--patches', type=Path, required=True)
    ap.add_argument('--front', type=float, default=240.)
    ap.add_argument('--count', type=int, default=200)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--drift', type=float, default=32.)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = F.load_cfg()
    pad = view_radius(cfg.model)+cfg.data.step+4+args.drift
    c = copy.deepcopy(cfg)
    c.data.patches_dir = [str(args.patches)]
    seeds, spec, data = draw_seeds(c, args.count, args.steps, args.front, .95, all_val=False, max_per_patch=0,
                                   seed_split=True)
    matrix = F.affine()
    import numpy as np
    scale9 = float(np.cbrt(abs(np.linalg.det(matrix[:3, :3]))))
    union2, union9, rows = set(), set(), []
    for w in seeds:
        k2 = F.step_keys(w, spec, data, args.steps, pad)
        k9 = F.step_keys(w, spec, data, args.steps, pad, matrix, scale9)
        c2, c9 = F.classify('l2', k2, True), F.classify('9362', k9, True)
        union2 |= k2
        union9 |= k9
        rows.append(dict(meta=w['meta'], missing_l2=c2['noncached']+c2['unlisted'],
                         missing_9362=c9['noncached']+c9['unlisted']))
    report = dict(front=args.front, count=args.count, drawn=len(seeds), pad=pad,
                  union_l2=F.classify('l2', union2, True), union_9362=F.classify('9362', union9, True),
                  first40_union=None, seeds=rows)
    first = seeds[:40]
    u2, u9 = set(), set()
    for w in first:
        u2 |= F.step_keys(w, spec, data, args.steps, pad)
        u9 |= F.step_keys(w, spec, data, args.steps, pad, matrix, scale9)
    report['first40_union'] = dict(l2=F.classify('l2', u2, True), x9362=F.classify('9362', u9, True))
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: report[k] for k in ('drawn', 'union_l2', 'union_9362', 'first40_union')}))


if __name__ == '__main__':
    main()
