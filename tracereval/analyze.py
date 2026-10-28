"""Seed-bootstrap intervals, the pre-registered GO / NO-GO rule, and paired comparisons for benchmark runs.

A run folder holds villa's ``report.json`` plus ``run_meta.json`` and ``per_trace.json`` from ``tracereval.run_bench``.
Pooled metrics are recomputed from the per-trace counts exactly as ``benchmark.summarize`` pools them (per step:
sums over traces, then the mean over steps) and checked against the report before any interval is drawn.

  python -m tracereval.analyze summary RUN [RUN ...]
  python -m tracereval.analyze gate --team 0.457,0.980 0.605,0.980 0.548,0.957 --runs RUN5 RUN9 RUN15
  python -m tracereval.analyze paired REF RUN [RUN ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COVERAGE_MARGIN = .12
PRECISION_MARGIN = .04
MIN_SEEDS = 40


def load(run: Path, traces_file: str = 'per_trace.json') -> dict:
    report = json.loads((run/'report.json').read_text())
    traces = json.loads((run/traces_file).read_text())
    meta = json.loads((run/'run_meta.json').read_text())
    if len(report['models']) != 1:
        raise ValueError(f'{run}: one model per run expected')
    name, entry = next(iter(report['models'].items()))
    return dict(run=run, name=name, report=report, summary=entry['summary'], traces=traces, meta=meta,
                seeds=report['seeds'], front=report['front'], steps=report['steps'])


def pooled(traces: list, index=None) -> dict:
    index = np.arange(len(traces)) if index is None else index
    arr = lambda key: np.asarray([traces[i][key] for i in index], np.float64)
    ok, kept, cover = arr('ok').sum(0), arr('kept').sum(0), arr('cover').sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        coverage = np.where(ok > 0, cover/np.maximum(ok, 1), np.nan)
        precision = np.where(kept > 0, cover/np.maximum(kept, 1), np.nan)
    return dict(coverage=float(np.nanmean(coverage)), precision=float(np.nanmean(precision)),
                real_switches=int(sum(traces[i]['first_real_switch'] is not None for i in index)),
                n=len(index))


def bootstrap(traces: list, n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    draws = [pooled(traces, rng.integers(0, len(traces), len(traces))) for _ in range(n_boot)]
    ci = lambda key: [float(np.percentile([d[key] for d in draws], 2.5)), float(np.percentile([d[key] for d in draws], 97.5))]
    return dict(coverage=ci('coverage'), precision=ci('precision'))


def summary(run: Path) -> dict:
    r = load(run)
    p = pooled(r['traces'])
    s = r['summary']
    if abs(p['coverage']-s['coverage_mean']) > 1e-9 or abs(p['precision']-s['precision_mean']) > 1e-9:
        raise ValueError(f"{run}: per-trace counts do not reproduce the report ({p} vs {s['coverage_mean']}, "
                         f"{s['precision_mean']})")
    missing = r['meta'].get('missing_chunks_total', {})
    return dict(run=str(run), seeds=len(r['seeds']), traces=p['n'], front=r['front'],
                points=int(round(r['front']/60.))+1, steps=r['steps'], coverage=p['coverage'],
                precision=p['precision'], real_switches=p['real_switches'], lost=s['traces_lost'],
                stopped=s['traces_stopped'], kept=s['kept_mean'], same_sheet=s['same_sheet_mean'],
                caught=s['caught_mean'], dist_p50=s['dist_p50_mean'], ci=bootstrap(r['traces']),
                noncached_reads=int(missing.get('noncached', 0)), unlisted_reads=int(missing.get('unlisted', 0)),
                wall_seconds=r['meta']['wall_seconds'])


def gate(team: list, runs: list) -> dict:
    """The rule of PLAN.md section 5 (integrity and regime clauses; the MPS clause is checked separately)."""
    rows, reasons = [], []
    for (coverage_team, precision_team), run in zip(team, runs):
        s = summary(run)
        row = dict(s, team_coverage=coverage_team, team_precision=precision_team,
                   coverage_floor=round(coverage_team-COVERAGE_MARGIN, 3),
                   precision_floor=round(precision_team-PRECISION_MARGIN, 3))
        row['integrity'] = s['noncached_reads'] == 0 and s['unlisted_reads'] == 0 and s['seeds'] >= MIN_SEEDS
        row['coverage_ok'] = s['coverage'] >= coverage_team-COVERAGE_MARGIN
        row['precision_ok'] = s['precision'] >= precision_team-PRECISION_MARGIN
        row['above_band'] = s['coverage'] > coverage_team+COVERAGE_MARGIN
        if not row['integrity']:
            reasons.append(f"{s['points']} points: integrity (seeds {s['seeds']}, noncached {s['noncached_reads']}, "
                           f"unlisted {s['unlisted_reads']})")
        if not row['coverage_ok']:
            reasons.append(f"{s['points']} points: coverage {s['coverage']:.3f} below {row['coverage_floor']}")
        if not row['precision_ok']:
            reasons.append(f"{s['points']} points: precision {s['precision']:.3f} below {row['precision_floor']}")
        rows.append(row)
    return dict(decision='GO' if not reasons else 'NO-GO', reasons=reasons, rows=rows)


def paired(ref: Path, run: Path, n_boot: int = 2000, seed: int = 0, run_traces: str = 'per_trace.json') -> dict:
    a, b = load(ref), load(run, run_traces)
    # Patch name, orientation and anchor identify a seed; the patch directory differs between model frames
    # (C1 to C3 read the same patches written into their own frame), so it is not part of the key.
    key = lambda s: (s['patch'], s['axis'], s['sign_u'], s['sign_v'], tuple(round(v, 6) for v in s['anchor']))
    ka = {key(s): i for i, s in enumerate(a['seeds'])}
    pairs = [(ka[key(s)], j) for j, s in enumerate(b['seeds']) if key(s) in ka]
    if not pairs:
        raise ValueError('no common seeds')
    ia, ib = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    base, other = pooled(a['traces'], ia), pooled(b['traces'], ib)
    rng = np.random.default_rng(seed)
    deltas = {'coverage': [], 'precision': []}
    for _ in range(n_boot):
        pick = rng.integers(0, len(pairs), len(pairs))
        pa, pb = pooled(a['traces'], ia[pick]), pooled(b['traces'], ib[pick])
        for k in deltas:
            deltas[k].append(pb[k]-pa[k])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in deltas.items()}
    return dict(ref=str(ref), run=str(run), pairs=len(pairs), ref_only=len(a['seeds'])-len(pairs),
                ref_metrics=base, run_metrics=other,
                change=dict(coverage=other['coverage']-base['coverage'], precision=other['precision']-base['precision']),
                ci=ci)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('summary')
    s.add_argument('runs', nargs='+', type=Path)
    g = sub.add_parser('gate')
    g.add_argument('--team', nargs='+', required=True, help='coverage,precision per run, in run order')
    g.add_argument('--runs', nargs='+', type=Path, required=True)
    p = sub.add_parser('paired')
    p.add_argument('ref', type=Path)
    p.add_argument('runs', nargs='+', type=Path)
    ap.add_argument('--json', type=Path)
    args = ap.parse_args(argv)
    if args.cmd == 'summary':
        out = [summary(r) for r in args.runs]
    elif args.cmd == 'gate':
        team = [tuple(float(v) for v in t.split(',')) for t in args.team]
        out = gate(team, args.runs)
    else:
        out = [paired(args.ref, r) for r in args.runs]
    text = json.dumps(out, indent=1)
    if args.json:
        args.json.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
