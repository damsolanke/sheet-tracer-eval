"""rescore_tol on a two-trace run, with small stand-ins for the two villa functions it calls.

The stand-in ``landing`` measures the distance to the nearest labelled point and, like villa's, calls a point on the
sheet when that distance is below the tolerance. rescore_tol runs in a subprocess because ``env.setup`` changes
process-wide state (environment variables, ``sys.modules['vesuvius']``, torch threads).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]

STUB_LOSSES = '''
import torch


def landing(points, gt, valid, tolerance):
    labelled = gt[0][valid[0]].double()
    dist = torch.cdist(points[0].double(), labelled).min(1).values
    ok = torch.ones_like(dist, dtype=torch.bool)
    return None, dist[None], (ok & (dist < tolerance))[None], ok[None]
'''

STUB_BENCHMARK = '''
import numpy as np


def _read_xyz(path):
    return np.load(path/'xyz.npy')
'''

# Offsets (voxels) of each traced point from the labelled sheet; None = not kept (written as -1).
OFFSETS = [
    [[0., 2., 5.9, 6.3, None], [6.6, 6.7, 1., 1., 1.], [None]*5, [None]*5],
    [[3., None, 6.05, 7.5, 0.5], [None, 6.2, 6.68, 6.69, 2.], [1., 1., 1., 1., 1.], [5.99, 6., 6.01, 6.68, 6.7]],
]


def cover(offsets, tolerance):
    return [sum(o is not None and abs(o) < tolerance for o in row) for row in offsets]


def villa_stub(root: Path) -> Path:
    sheets = root/'vesuvius'/'neural_tracing'/'tracers'/'sheets'
    (sheets/'evaluation').mkdir(parents=True)
    for d in (root/'vesuvius'/'neural_tracing', root/'vesuvius'/'neural_tracing'/'tracers', sheets,
              sheets/'evaluation'):
        (d/'__init__.py').write_text('')
    (sheets/'losses.py').write_text(STUB_LOSSES)
    (sheets/'evaluation'/'benchmark.py').write_text(STUB_BENCHMARK)
    return root


def make_run(folder: Path, base_cover=None, held_out=False) -> Path:
    gt = np.stack(np.meshgrid(np.arange(5)*10., np.arange(6)*10., indexing='xy'), -1)  # (6, 5, 2): x, y
    gt = np.concatenate([gt, np.full(gt.shape[:2]+(1,), 100.)], -1).astype(np.float32)  # sheet at z = 100
    traces = []
    for i, offsets in enumerate(OFFSETS):
        rows = np.full((len(offsets)+1, 5, 3), -1., np.float32)  # row 0 is the seed row, dropped by rescore_tol
        for r, row in enumerate(offsets, start=1):
            for c, o in enumerate(row):
                if o is not None:
                    rows[r, c] = gt[r, c]+np.array([0., 0., o], np.float32)
        for name, xyz in ((f'seed{i:02d}_gt', gt), (f'seed{i:02d}_tracer7', rows)):
            (folder/'tifxyz'/name).mkdir(parents=True)
            np.save(folder/'tifxyz'/name/'xyz.npy', xyz)
        counts = cover(offsets, 6.) if base_cover is None else base_cover[i]
        traces.append(dict(ok=[5]*4, kept=[sum(o is not None for o in row) for row in offsets], cover=counts,
                           seen=[5]*4, first_real_switch=None, stopped=None, off_kept_points=0))
    report = dict(models={'tracer7': {}}, seeds=[{}, {}], steps=4, front=240.)
    if held_out:
        report['held_out_scoring'] = True
    (folder/'report.json').write_text(json.dumps(report))
    (folder/'per_trace.json').write_text(json.dumps(traces))
    return folder


def rescore(tmp_path: Path, run: Path, tolerance: str):
    env = dict(os.environ, VILLA_SRC=str(villa_stub(tmp_path/'villa_src')), TRACEREVAL_DATA=str(tmp_path/'data'),
               VESUVIUS_DATA=str(tmp_path/'vesuvius'))
    return subprocess.run([sys.executable, '-m', 'tracereval.rescore_tol', str(run), '--tolerance', tolerance],
                          cwd=ROOT, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)


def test_rescore_reproduces_the_run_then_rescores_at_the_new_tolerance(tmp_path):
    run = make_run(tmp_path/'run')
    proc = rescore(tmp_path, run, '6.6854')
    assert proc.returncode == 0, proc.stderr
    out = json.loads((run/'per_trace_tol6.6854.json').read_text())
    before = json.loads((run/'per_trace.json').read_text())
    for i, offsets in enumerate(OFFSETS):
        assert out[i]['cover'] == cover(offsets, 6.6854)
        assert {k: v for k, v in out[i].items() if k != 'cover'} == {k: v for k, v in before[i].items() if k != 'cover'}
    # The two traces gain 1 + 1 and 1 + 2 + 0 + 3 on-sheet points between 6 and 6.6854 voxels.
    assert [sum(t['cover'])-sum(b['cover']) for t, b in zip(out, before)] == [2, 6]


def test_rescore_refuses_a_run_whose_counts_it_cannot_reproduce(tmp_path):
    wrong = [cover(o, 6.) for o in OFFSETS]
    wrong[1][2] -= 1
    run = make_run(tmp_path/'run', base_cover=wrong)
    proc = rescore(tmp_path, run, '6.1698')
    assert proc.returncode != 0
    assert '1 traces do not reproduce their cover counts' in proc.stderr
    assert not (run/'per_trace_tol6.1698.json').exists()


def test_rescore_refuses_held_out_scoring_runs(tmp_path):
    proc = rescore(tmp_path, make_run(tmp_path/'run', held_out=True), '6.1698')
    assert proc.returncode != 0 and 'held-out scoring runs are not supported' in proc.stderr


def test_matched_tolerances_follow_from_the_condition_voxels():
    """The section 4 tolerance check: 6 voxels of C3 (9.627 um) expressed in C1 and C2 voxels."""
    from tracereval import grid
    voxel = {c: spec['voxel_um'] for c, spec in grid.CONDITIONS.items()}
    assert round(6*voxel['C3']/voxel['C1'], 4) == 6.1698
    assert round(6*voxel['C3']/voxel['C2'], 4) == 6.6854
    assert [round(6*voxel[c], 1) for c in ('C2', 'C1', 'C3')] == [51.8, 56.2, 57.8]
