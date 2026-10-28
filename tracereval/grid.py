"""PLAN.md sections 3 to 6 on the A0 slab: one ``run_bench`` call per (condition, front, threshold), in order.

Fixed settings (PLAN.md section 3): 40 seeds, 60 steps, ``--all-val``, ``--max-per-patch 0``, ``--min-valid 0.95``,
benchmark seed 0 (``draw_seeds``' default). C0 reads the 2.399 um scan at level 2 with the prepared patches; C1 to C3
read the 9.362 um scan with the patches written into their model frames (``tracereval.resample``) and reuse C0's
seeds at the same front (``--seeds-from``: same patch, orientation and anchor; only ``root`` points at the frame's
patch directory). A run whose ``run_meta.json`` exists is skipped unless ``--force``.

  python -m tracereval.grid --cond C0 --fronts 240 480 840 --thresholds 0.4 --device mps --threads 2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('TRACEREVAL_DATA', '~/kaggle-data/vesuvius/tracereval')).expanduser()
CKPT = Path(os.environ.get('VESUVIUS_DATA', '~/kaggle-data/vesuvius')).expanduser() / 'hf' / 'sheet_tracer_7pt_gap60' / 'last.pt'
PATCHES = DATA/'prepared'/'PHerc0139_2399l2_A0slab'/'heldout_patches'
AFFINE = ROOT.parent/'windeval'/'results'/'transform_pherc0139_2399l2_to_9362.json'
VOL_L2 = DATA/'volumes'/'PHerc0139_2399um_l2.zarr'
VOL_9362 = DATA/'volumes'/'PHerc0139_9362um.zarr'
# PLAN.md section 4: s maps the 9.362 um scan into the model frame (x_model = s * x_9362).
CONDITIONS = {
    'C0': dict(scale=None, volume=VOL_L2, voxel_um=9.596),
    'C1': dict(scale=1., volume=VOL_9362, voxel_um=9.362),
    'C2': dict(scale=9.362/8.64, volume=VOL_9362, voxel_um=8.64),
    'C3': dict(scale=1/1.0283, volume=VOL_9362, voxel_um=9.362*1.0283),
}


def run_name(cond: str, front: float, threshold: float) -> str:
    return f'A0_{cond}_f{int(front)}_t{threshold:g}'


def frame_patches(cond: str) -> Path:
    """The A0 patches in the condition's model frame (written once)."""
    spec = CONDITIONS[cond]
    if spec['scale'] is None:
        return PATCHES
    dst = DATA/'prepared'/f'PHerc0139_A0slab_{cond}_frame'
    if not (dst/'frame.json').is_file():
        subprocess.run([sys.executable, '-m', 'tracereval.resample', str(PATCHES), str(dst), '--affine', str(AFFINE),
                        '--scale', repr(spec['scale'])], cwd=ROOT, check=True)
    return dst


def seeds_file(cond: str, front: float, patches: Path, seed_run: str) -> Path:
    """C0's seeds at this front with ``root`` set to the frame's patch directory."""
    report = json.loads((DATA/'runs'/seed_run/'report.json').read_text())
    seeds = [dict(meta, root=str(patches)) for meta in report['seeds']]
    out = DATA/'runs'/'seeds'/f'{cond}_f{int(front)}_from_{seed_run}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(seeds=seeds, source=str(DATA/'runs'/seed_run/'report.json')), indent=1))
    return out


def run(cond: str, front: float, threshold: float, device: str, threads: int, force: bool) -> dict:
    name = run_name(cond, front, threshold)
    out = DATA/'runs'/name
    meta_path = out/'run_meta.json'
    if meta_path.is_file() and not force:
        print(f'{name}: exists, skipped', flush=True)
        return json.loads(meta_path.read_text())
    spec = CONDITIONS[cond]
    patches = frame_patches(cond)
    wrapper = ['--threads', str(threads), '--device', device, '--no-figures']
    bench = ['--checkpoints', str(CKPT), '--names', 'tracer7', '--patches-dir', str(patches), '--all-val',
             '--max-per-patch', '0', '--min-valid', '0.95', '--seeds', '40', '--steps', '60', '--front', f'{front:g}',
             '--threshold', f'{threshold:g}', '--volume', str(spec['volume']), '--voxel-size', f"{spec['voxel_um']:.4f}",
             '--out', str(out)]
    if spec['scale'] is not None:
        bench += ['--seeds-from', str(seeds_file(cond, front, patches, run_name('C0', front, .4)))]
        if spec['scale'] != 1.:
            spec_path = DATA/'runs'/'seeds'/f'{cond}_resample.json'
            spec_path.write_text(json.dumps(dict(affine=str(AFFINE), scale=spec['scale'])))
            wrapper += ['--resample', str(spec_path)]
    log = DATA/'logs'/f'{name}.log'
    t0 = time.time()
    print(f"{time.strftime('%H:%M:%S')} {name}: start", flush=True)
    with open(log, 'w') as handle:
        rc = subprocess.run(['nice', '-n', '15', sys.executable, '-m', 'tracereval.run_bench', *wrapper, '--', *bench],
                            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not meta_path.is_file():
        raise SystemExit(f'{name}: rc {rc}, see {log}')
    meta = json.loads(meta_path.read_text())
    report = json.loads((out/'report.json').read_text())
    s = report['models']['tracer7']['summary']
    print(f"{time.strftime('%H:%M:%S')} {name}: {len(report['seeds'])} seeds, coverage {s['coverage_mean']:.3f} "
          f"precision {s['precision_mean']:.4f} real switches {s['traces_real_switch']}, missing chunks "
          f"{meta['missing_chunks_total']}, {time.time()-t0:.0f} s", flush=True)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cond', nargs='+', required=True, choices=sorted(CONDITIONS))
    ap.add_argument('--fronts', type=float, nargs='+', default=[240., 480., 840., 360.])
    ap.add_argument('--thresholds', type=float, nargs='+', default=[.4, .3, .5])
    ap.add_argument('--device', default='mps')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)
    for cond in args.cond:
        for front in args.fronts:
            for threshold in args.thresholds:
                run(cond, front, threshold, args.device, args.threads, args.force)


if __name__ == '__main__':
    main()
