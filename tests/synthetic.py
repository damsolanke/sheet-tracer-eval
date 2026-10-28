"""Tiny benchmark run folders in the layout tracereval.run_bench writes (report.json, per_trace.json, run_meta.json).

The report's pooled summary is computed here with plain loops, independently of tracereval.analyze, so the tests
also check analyze's pooling against a second implementation.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path


def pooled_by_hand(traces: list) -> tuple[float, float]:
    steps = len(traces[0]['ok'])
    cov, prec = [], []
    for s in range(steps):
        ok = sum(t['ok'][s] for t in traces)
        kept = sum(t['kept'][s] for t in traces)
        cover = sum(t['cover'][s] for t in traces)
        if ok:
            cov.append(cover/ok)
        if kept:
            prec.append(cover/kept)
    mean = lambda xs: sum(xs)/len(xs) if xs else math.nan
    return mean(cov), mean(prec)


def trace(ok, kept, cover, seen=None, stopped=None, real_switch=None) -> dict:
    seen = list(ok) if seen is None else list(seen)
    return dict(ok=list(ok), same=list(cover), seen=seen, kept=list(kept), cover=list(cover), off=[0]*len(ok),
                caught=[0]*len(ok), first_switch=real_switch, first_real_switch=real_switch, off_kept_points=0,
                lost=None, stopped=stopped, dist_median=[1.]*len(ok))


def random_traces(n: int, steps: int, seed: int) -> list:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        traced = rng.choice([steps, steps, rng.randint(0, steps-1)])
        ok = [rng.randint(5, 9) for _ in range(steps)]
        kept = [rng.randint(0, o) if s < traced else 0 for s, o in enumerate(ok)]
        cover = [rng.randint(0, k) for k in kept]
        seen = [o if s < traced else 0 for s, o in enumerate(ok)]
        out.append(trace(ok, kept, cover, seen, stopped=None if traced == steps else traced+1,
                         real_switch=rng.choice([None, None, rng.randint(1, steps)])))
    return out


def seed_meta(i: int, root: str = '/data/patches') -> dict:
    return dict(patch=f'patch{i % 7:02d}', root=root, axis=i % 2, sign_u=1, sign_v=-1 if i % 3 else 1,
                anchor=[10.5+i, 200.25+3*i])


def write_run(folder: Path, traces: list, front: float = 360., seeds: list | None = None, name: str = 'tracer7',
              noncached: int = 0, unlisted: int = 0) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    seeds = [seed_meta(i) for i in range(len(traces))] if seeds is None else seeds
    coverage, precision = pooled_by_hand(traces)
    summary = dict(coverage_mean=coverage, precision_mean=precision, kept_mean=.5, same_sheet_mean=.9,
                   caught_mean=.7, dist_p50_mean=1.05, traces_lost=0,
                   traces_stopped=sum(t['stopped'] is not None for t in traces))
    report = dict(checkpoints={name: '/ckpt/last.pt'}, steps=len(traces[0]['ok']), front=front, seeds=seeds,
                  models={name: dict(summary=summary)})
    missing = {'absent': 3}
    if noncached:
        missing['noncached'] = noncached
    if unlisted:
        missing['unlisted'] = unlisted
    meta = dict(wall_seconds=12.5, missing_chunks_total=missing,
                traces=[dict(steps_traced=sum(1 for x in t['seen'] if x) if t['stopped'] is not None
                             else len(t['ok'])) for t in traces])
    (folder/'report.json').write_text(json.dumps(report))
    (folder/'per_trace.json').write_text(json.dumps(traces))
    (folder/'run_meta.json').write_text(json.dumps(meta))
    return folder
