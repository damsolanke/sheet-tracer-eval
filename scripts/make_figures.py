"""Figures for README.md from the per-trace counts in results/ (numbers only; no CT or label data is drawn).

  python scripts/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'results'
FIGURES = ROOT/'figures'
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
FRONTS = (240, 360, 480, 840)
POINTS = {240: 5, 360: 7, 480: 9, 840: 15}
# One blue, light to dark with front length (an ordinal ramp: the front sizes are ordered).
COLOR = {240: '#86b6ef', 360: '#3987e5', 480: '#1c5cab', 840: '#0d366b'}
MARKER = {.3: 'o', .4: 's', .5: '^'}

plt.rcParams.update({
    'font.family': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'], 'font.size': 9,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.linewidth': .8, 'axes.labelcolor': INK2, 'axes.titlesize': 9.5,
    'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.titlecolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.size': 0, 'ytick.major.size': 0,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': .6, 'axes.axisbelow': True, 'legend.frameon': False, 'text.color': INK,
})


def load(run: str):
    traces = json.loads((RESULTS/run/'per_trace.json').read_text())
    meta = json.loads((RESULTS/run/'run_meta.json').read_text())
    return traces, meta


def coverage_per_step(traces: list) -> np.ndarray:
    """villa's per-step coverage: kept on-sheet points over scored points, summed over traces."""
    ok = np.sum([t['ok'] for t in traces], 0).astype(float)
    cover = np.sum([t['cover'] for t in traces], 0).astype(float)
    return np.where(ok > 0, cover/np.maximum(ok, 1), np.nan)


def running_per_step(meta: dict, steps: int) -> np.ndarray:
    """Traces that reached each step (rows traced before villa's stop rule ended them)."""
    traced = np.array([t['steps_traced'] for t in meta['traces']])
    return np.array([(traced >= s).sum() for s in range(1, steps+1)])


def end_labels(ax, ends: dict, x: float, min_gap: float):
    """Front-size labels at the right end of each line, nudged apart only where two would overlap."""
    order = sorted(ends, key=lambda f: ends[f])
    placed = []
    for f in order:
        y = ends[f] if not placed else max(ends[f], placed[-1]+min_gap)
        placed.append(y)
        ax.annotate(f'{POINTS[f]} points', (x, ends[f]), xytext=(x+1.2, y), textcoords='data', va='center',
                    ha='left', color=INK2, fontsize=8.5, annotation_clip=False,
                    arrowprops=dict(arrowstyle='-', color=AXIS, lw=.6, shrinkA=0, shrinkB=0) if y != ends[f] else None)


def figure_steps(threshold: float = .4) -> Path:
    fig, (ax_cov, ax_run) = plt.subplots(1, 2, figsize=(8.6, 3.3), sharex=True)
    cov_end, run_end = {}, {}
    for f in FRONTS:
        traces, meta = load(f'A0_C0_f{f}_t{threshold:g}')
        cov = coverage_per_step(traces)
        steps = np.arange(1, len(cov)+1)
        run = running_per_step(meta, len(cov))
        kw = dict(color=COLOR[f], lw=1.6, solid_capstyle='round', solid_joinstyle='round')
        ax_cov.plot(steps, cov, label=f'{POINTS[f]} points', **kw)
        ax_run.plot(steps, run, **kw)
        cov_end[f], run_end[f] = cov[-1], run[-1]
    ax_cov.set(ylim=(0, 1.02), xlim=(1, 60), xlabel='Step (20 voxels each)', ylabel='Coverage at the step')
    ax_cov.set_title('Coverage at each step')
    ax_run.set(ylim=(0, 41), xlabel='Step (20 voxels each)', ylabel='Traces (of 40)')
    ax_run.set_title('Traces still running')
    ax_cov.set_xticks([1, 10, 20, 30, 40, 50, 60])
    end_labels(ax_cov, cov_end, 60, .055)
    end_labels(ax_run, run_end, 60, 2.2)
    fig.legend(*ax_cov.get_legend_handles_labels(), loc='upper center', ncol=4, bbox_to_anchor=(.5, 1.0),
               handlelength=1.6, columnspacing=1.6, fontsize=8.5, labelcolor=INK2)
    fig.text(.01, .955, f'Threshold {threshold:g}, 40 seeds per front size', color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0, .94, .9), w_pad=4)
    out = FIGURES/'fig1_coverage_per_step.png'
    fig.savefig(out, dpi=170, metadata={'Software': None})
    plt.close(fig)
    return out


def figure_threshold() -> Path:
    table = json.loads((RESULTS/'A0_C0_table.json').read_text())
    rows = {(r['front'], r['threshold']): r for r in table['rows']}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for f in FRONTS:
        pts = [rows[(float(f), t)] for t in (.5, .4, .3)]
        ax.plot([p['precision'] for p in pts], [p['coverage'] for p in pts], color=COLOR[f], lw=1.6, zorder=2,
                solid_capstyle='round', solid_joinstyle='round')
        for p in pts:
            ax.plot(p['precision'], p['coverage'], MARKER[p['threshold']], color=COLOR[f], ms=7, mec=SURFACE,
                    mew=1.4, zorder=3)
        end = pts[0]
        ax.annotate(f'{POINTS[f]} points', (end['precision'], end['coverage']), xytext=(7, -1),
                    textcoords='offset points', va='center', ha='left', color=INK2, fontsize=8.5)
    ax.set(xlim=(.9, 1.0), ylim=(0, .8), xlabel='Precision (on-sheet share of kept points)',
           ylabel='Coverage (mean over 60 steps)')
    ax.set_title('Coverage against precision for each front size and threshold')
    fronts = [Line2D([], [], color=COLOR[f], lw=1.6, label=f'{POINTS[f]} points') for f in FRONTS]
    marks = [Line2D([], [], ls='', marker=MARKER[t], color=MUTED, ms=7, mec=SURFACE, mew=1.4,
                    label=f'{t:g}' + {.3: ' (suggested in villa)', .4: ' (team table)', .5: ' (stored default)'}[t])
             for t in (.3, .4, .5)]
    legend = dict(loc='lower left', fontsize=8.5, title_fontsize=8.5, labelcolor=INK2, alignment='left')
    ax.add_artist(ax.legend(handles=fronts, title='Front size', bbox_to_anchor=(0, .3), handlelength=1.6, **legend))
    ax.legend(handles=marks, title='Threshold', bbox_to_anchor=(0, 0), **legend)
    fig.tight_layout()
    out = FIGURES/'fig2_threshold_tradeoff.png'
    fig.savefig(out, dpi=170, metadata={'Software': None})
    plt.close(fig)
    return out


STOCK, PIECES = '#2a78d6', '#eb6834'


def figure_long_rows() -> Path:
    """Coverage by seed-row length at threshold 0.4: villa's trace against 7-point pieces (README section 7)."""
    import sys
    sys.path.insert(0, str(ROOT))
    from tracereval import analyze
    cov = lambda run: analyze.pooled(analyze.load(run)['traces'])['coverage']
    rows = {POINTS[f]: (cov(RESULTS/f'A0_C0_f{f}_t0.4'), cov(RESULTS/'long_rows'/f'A0_C0_f{f}_t0.4_pieces7s2'), True)
            for f in (360, 480, 840)}
    lengths = json.loads((RESULTS/'long_rows'/'A0_pieces_table.json').read_text())['exploratory_lengths']
    for n, key in ((13, '13_all'), (21, '21_clean')):
        rows[n] = (lengths[key]['stock']['coverage'], lengths[key]['pieces']['coverage'], False)
    xs = sorted(rows)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for i, (name, color) in enumerate((('villa trace', STOCK), ('7-point pieces', PIECES))):
        ys = [rows[x][i] for x in xs]
        ax.plot(xs, ys, color=color, lw=1.6, zorder=2, solid_capstyle='round', solid_joinstyle='round')
        for x, y in zip(xs, ys):
            filled = rows[x][2]
            ax.plot(x, y, 'o', ms=8, zorder=3, color=color, mfc=color if filled else SURFACE,
                    mec=SURFACE if filled else color, mew=1.4 if filled else 1.6)
        ax.annotate(name, (xs[-1], ys[-1]), xytext=(8, 0), textcoords='offset points', va='center', ha='left',
                    color=INK2, fontsize=8.5, annotation_clip=False)
    ax.set(xlim=(6, 22), ylim=(0, .8), xlabel='Seed-row length (points)', ylabel='Coverage (mean over 60 steps)')
    ax.set_xticks(xs)
    ax.set_title('Coverage by seed-row length at threshold 0.4')
    handles = [Line2D([], [], color=STOCK, lw=1.6, label='villa trace'),
               Line2D([], [], color=PIECES, lw=1.6, label='7-point pieces'),
               Line2D([], [], ls='', marker='o', ms=8, color=MUTED, mec=SURFACE, mew=1.4, label='released test seeds (40)'),
               Line2D([], [], ls='', marker='o', ms=8, color=MUTED, mfc=SURFACE, mew=1.6,
                      label='new windows, exploratory (40; 27 at 21 points)')]
    ax.legend(handles=handles, loc='lower left', fontsize=8.5, labelcolor=INK2, handlelength=1.6)
    fig.tight_layout(rect=(0, 0, .9, 1))
    out = FIGURES/'fig3_long_rows.png'
    fig.savefig(out, dpi=170, metadata={'Software': None})
    plt.close(fig)
    return out


if __name__ == '__main__':
    FIGURES.mkdir(exist_ok=True)
    for path in (figure_steps(), figure_threshold(), figure_long_rows()):
        print(path.relative_to(ROOT), f'{path.stat().st_size/1024:.0f} KiB')
