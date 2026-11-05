"""Trace a long seed row as overlapping pieces of the front size the checkpoint was trained on.

``scrollprize/sheet_tracer_7pt_gap60`` was trained on 7-point fronts. On the held-out slab at threshold 0.4 its
coverage is 0.54 / 0.64 at 7 / 9 points but 0.40 at 15 (sheet-tracer-eval results). ``piece_trace`` wraps villa's
``inference.trace``: a seed of more than ``piece`` points is cut into pieces of ``piece`` points starting every
``stride`` points (the last piece ends at the last point), each piece is traced on its own with the same options,
and the rows are merged column by column. Dense column j of the full front (``spline_samples`` per gap) comes, at
each step, from the piece that is still tracing at that step and holds j closest to its own centre; a column no
piece covers any more keeps its last position with confidence 0, as the benchmark pads a stopped trace. The merged
trace is as long as the longest piece. Seeds of ``piece`` points or fewer go to ``trace`` unchanged.
"""
from __future__ import annotations

import numpy as np


def starts(points: int, piece: int, stride: int) -> list[int]:
    if points <= piece:
        return [0]
    out = list(range(0, points-piece+1, stride))
    if out[-1] != points-piece:
        out.append(points-piece)
    return out


def merge(outs: list, offsets: list[int], columns: int) -> tuple[np.ndarray, np.ndarray]:
    """Merge per-piece (rows (T_k+1, C_k, 3), confs (T_k+1, C_k)) placed at column offsets into one trace."""
    T = max(len(r) for r, _ in outs)
    rows = np.zeros((T, columns, 3), outs[0][0].dtype)
    confs = np.zeros((T, columns), outs[0][1].dtype)
    best = np.full((T, columns), np.inf)
    for (r, c), off in zip(outs, offsets):
        width = r.shape[1]
        cols = np.arange(off, off+width)
        central = np.abs(np.arange(width)-(width-1)/2.)
        for t in range(len(r)):
            take = central < best[t, cols]
            rows[t, cols[take]] = r[t, take]
            confs[t, cols[take]] = c[t, take]
            best[t, cols[take]] = central[take]
    for t in range(1, T):
        gone = ~np.isfinite(best[t])
        rows[t, gone] = rows[t-1, gone]
        confs[t, gone] = 0
    if not np.all(np.isfinite(best[0])):
        raise ValueError('the pieces do not cover every column of the seed row')
    return rows, confs


def piece_trace(trace, piece: int = 7, stride: int = 4, samples: int = 6):
    def traced(model, cfg, seed, direction, steps, device, **kw):
        n = len(seed)
        if n <= piece:
            return trace(model, cfg, seed, direction, steps, device, **kw)
        first = starts(n, piece, stride)
        outs = [trace(model, cfg, seed[s:s+piece], direction[s:s+piece], steps, device, **kw) for s in first]
        if cfg.trace.spline_samples != samples:
            raise ValueError(f'spline_samples {cfg.trace.spline_samples} != {samples}')
        return merge(outs, [s*samples for s in first], (n-1)*samples+1)
    traced.piece, traced.stride = piece, stride
    return traced
