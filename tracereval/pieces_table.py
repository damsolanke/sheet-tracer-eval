"""Tables of the long-row section (README section 7) from the per-trace counts in results/long_rows/.

Decision (PLAN_PIECES.md section 4): 15-point fronts at threshold 0.4, pieces (piece 7, stride 2) against stock on the
same seeds, the 40 released test seeds pooled with the fresh windows. A seed whose run in either arm added a noncached
or unlisted chunk read (run_meta.json, per trace) is dropped from its pair. Paired bootstrap over pairs, 2,000
resamples, generator seed 0 (as ``analyze.paired``). Also the 7-point identity, the 9-point scope check, the reported
threshold 0.3 comparison, the tuning exploration and the exploratory row lengths.

  python -m tracereval.pieces_table --results results --out results/long_rows/A0_pieces_table.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tracereval import analyze


def _key(s):
    return (s['patch'], s['axis'], s['sign_u'], s['sign_v'], tuple(round(v, 6) for v in s['anchor']))


def _flagged(run: Path) -> set:
    meta = json.loads((run/'run_meta.json').read_text())
    return {i for i, t in enumerate(meta['traces'])
            if (t.get('new_missing_chunks') or {}).get('noncached', 0) or (t.get('new_missing_chunks') or {}).get('unlisted', 0)}


def pairs(ref: Path, run: Path, drop: bool = True):
    a, b = analyze.load(ref), analyze.load(run)
    ka = {_key(s): i for i, s in enumerate(a['seeds'])}
    pr = [(ka[_key(s)], j) for j, s in enumerate(b['seeds']) if _key(s) in ka]
    bad_a, bad_b = (_flagged(ref), _flagged(run)) if drop else (set(), set())
    kept = [(i, j) for i, j in pr if i not in bad_a and j not in bad_b]
    return a['traces'], b['traces'], kept, len(pr)


def stats(ta, tb, pr, n_boot: int = 2000, seed: int = 0) -> dict:
    ia, ib = np.array([p[0] for p in pr]), np.array([p[1] for p in pr])
    base, other = analyze.pooled(ta, ia), analyze.pooled(tb, ib)
    rng = np.random.default_rng(seed)
    d = {'coverage': [], 'precision': []}
    for _ in range(n_boot):
        pick = rng.integers(0, len(pr), len(pr))
        pa, pb = analyze.pooled(ta, ia[pick]), analyze.pooled(tb, ib[pick])
        for k in d:
            d[k].append(pb[k]-pa[k])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in d.items()}
    sw = lambda t, idx: int(sum(t[i]['first_real_switch'] is not None for i in idx))
    return dict(n=len(pr), stock=dict(coverage=base['coverage'], precision=base['precision'], real_switches=sw(ta, ia)),
                pieces=dict(coverage=other['coverage'], precision=other['precision'], real_switches=sw(tb, ib)),
                change=dict(coverage=other['coverage']-base['coverage'], precision=other['precision']-base['precision']),
                ci=ci)


def build(results: Path) -> dict:
    L = results/'long_rows'
    out = dict(identity_7=(L/'A0_C0_f360_t0.4_pieces7s2'/'per_trace.json').read_bytes()
               == (results/'A0_C0_f360_t0.4'/'per_trace.json').read_bytes())
    ra, rb, rp, rn = pairs(results/'A0_C0_f840_t0.4', L/'A0_C0_f840_t0.4_pieces7s2')
    fa, fb, fp, fn = pairs(L/'FRESH_f840_stock_t0.4', L/'FRESH_f840_t0.4_pieces7s2')
    pooled = rp+[(i+len(ra), j+len(rb)) for i, j in fp]
    out['decision_15'] = dict(pooled=stats(ra+fa, rb+fb, pooled), released=stats(ra, rb, rp), fresh=stats(fa, fb, fp),
                              dropped=dict(released=rn-len(rp), fresh=fn-len(fp)))
    a9, b9, p9, _ = pairs(results/'A0_C0_f480_t0.4', L/'A0_C0_f480_t0.4_pieces7s2')
    out['scope_9'] = stats(a9, b9, p9)
    a3, b3, p3, _ = pairs(results/'A0_C0_f840_t0.3', L/'A0_C0_f840_t0.3_pieces7s2')
    out['reported_15_t03'] = stats(a3, b3, p3)
    tune = {}
    for name, ref in (('pieces7s4_t0.4', 'stock_t0.4'), ('pieces7s4_t0.3', 'stock_t0.3'), ('pieces7s2_t0.4', 'stock_t0.4'),
                      ('pieces7s6_t0.4', 'stock_t0.4')):
        ta, tb, tp, _ = pairs(L/f'TUNE_f840_{ref}', L/f'TUNE_f840_{name}', drop=False)
        tune[name] = stats(ta, tb, tp)
    out['tuning_15'] = tune
    lengths = {}
    for f, pts in ((720, 13), (1200, 21)):
        for drop in (False, True):
            ta, tb, tp, n = pairs(L/f'EXPL_f{f}_stock_t0.4', L/f'EXPL_f{f}_pieces7s2_t0.4', drop=drop)
            lengths[f'{pts}_{"clean" if drop else "all"}'] = dict(stats(ta, tb, tp), pairs_before_drop=n)
    out['exploratory_lengths'] = lengths
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', type=Path, default=Path('results'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    out = build(args.results)
    args.out.write_text(json.dumps(out, indent=1))
    d = out['decision_15']
    for k in ('pooled', 'released', 'fresh'):
        s = d[k]
        print(f"15 points {k:>8} n {s['n']:>3}: coverage {s['stock']['coverage']:.3f} -> {s['pieces']['coverage']:.3f} "
              f"({s['change']['coverage']:+.3f} [{s['ci']['coverage'][0]:+.3f}, {s['ci']['coverage'][1]:+.3f}]), precision "
              f"{s['change']['precision']:+.4f}, switches {s['stock']['real_switches']} -> {s['pieces']['real_switches']}")
    print('identity at 7 points:', out['identity_7'])


if __name__ == '__main__':
    main()
