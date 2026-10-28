"""Which planned benchmarks fit the cached CT, and which chunks a later step would have to download.

Reads patches and the checkpoint config only; no CT is read and nothing is downloaded, except S3 *listings* of chunk
rows (``--list-bucket``: which chunk keys exist, a few kB per row, no chunk data) so that byte counts are exact.

Seeds are drawn with villa's own ``evaluation.benchmark.draw_seeds`` (as ``benchmark.main`` draws them). A trace reads
the CT within ``view_radius + step + 4`` of its front each step (``tracing.inference.trace``); the estimate takes each
seed's GT reach box (the supervised window over all steps) widened by that pad plus ``--drift`` voxels for traces
that wander off the GT sheet. The mirror estimate follows ``scripts/prepare_scroll.py``: every chunk within ``--pad``
(96) voxels of a kept node, at the level the model reads.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import urllib.parse
from pathlib import Path

import numpy as np

from tracereval import env

env.setup(threads=1)

import torch  # noqa: E402

from tracereval import chunks  # noqa: E402
from vesuvius.neural_tracing.tracers.sheets import config as C  # noqa: E402
from vesuvius.neural_tracing.tracers.sheets.evaluation.benchmark import draw_seeds  # noqa: E402
from vesuvius.neural_tracing.tracers.sheets.sampling import view_radius  # noqa: E402

BUCKET = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'
VOLUMES = {
    'l2': dict(remote='PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr/2',
               local=env.DATA/'volumes'/'PHerc0139_2399um_l2.zarr'/'0'),
    '9362': dict(remote='PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr/0',
                 local=env.DATA/'volumes'/'PHerc0139_9362um.zarr'/'0'),
}
AFFINE = Path(__file__).resolve().parents[2]/'windeval'/'results'/'transform_pherc0139_2399l2_to_9362.json'


def load_cfg():
    checkpoint = torch.load(env.CHECKPOINT, map_location='cpu', weights_only=True)
    return C.from_dict(checkpoint['config'], strict=False)


def affine() -> np.ndarray:
    m = np.eye(4)
    m[:3] = np.asarray(json.loads(AFFINE.read_text())['matrix_source_to_target_xyz'])
    return m


def list_row(remote: str, cz: int) -> set:
    """Chunk keys 'cy/cx' that exist in bucket row ``cz`` (S3 ListObjectsV2; cached in the catalog folder)."""
    import urllib.request
    cache = env.DATA/'catalog'/'listings'/remote.replace('/', '__')/f'{cz}.txt'
    if cache.is_file():
        return set(cache.read_text().split())
    prefix = f'{remote}/{cz}/'
    present, token = set(), None
    while True:
        url = f'{BUCKET}/?list-type=2&prefix={urllib.parse.quote(prefix)}'
        if token:
            url += '&continuation-token='+urllib.parse.quote(token)
        with urllib.request.urlopen(url, timeout=120) as response:
            text = response.read().decode()
        present.update(key[len(prefix):] for key in re.findall(r'<Key>([^<]*)</Key>', text))
        match = re.search(r'<NextContinuationToken>([^<]*)<', text)
        if not match:
            break
        token = match.group(1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text('\n'.join(sorted(present)))
    return present


def classify(volume: str, keys: set, list_bucket: bool) -> dict:
    """Counts of keys by class (cached / absent / noncached / unlisted), using our own listings for unlisted rows."""
    info = VOLUMES[volume]
    out = dict(cached=0, absent=0, noncached=0, unlisted=0)
    rows = {}
    for key in keys:
        kind = chunks.classify(info['local'], key)
        if kind == 'unlisted' and list_bucket:
            cz = key[0]
            if cz not in rows:
                rows[cz] = list_row(info['remote'], cz)
            kind = 'noncached' if f'{key[1]}/{key[2]}' in rows[cz] else 'absent'
        out[kind] += 1
    out['download_gib'] = round((out['noncached']+out['unlisted'])*chunks.CHUNK_BYTES/2**30, 2)
    return out


def box_keys(lo_xyz, hi_xyz, pad: float, matrix=None, scale_pad: float = 1.) -> set:
    lo, hi = np.asarray(lo_xyz, np.float64), np.asarray(hi_xyz, np.float64)
    if matrix is not None:
        corners = np.array([np.where(c, hi, lo) for c in np.ndindex(2, 2, 2)])
        mapped = corners@matrix[:3, :3].T+matrix[:3, 3]
        lo, hi = mapped.min(0), mapped.max(0)
        pad = pad*scale_pad
    return chunks.keys_for_box((lo-pad)[::-1], (hi+pad)[::-1])


def step_keys(w, spec, data, steps: int, pad: float, matrix=None, scale_pad: float = 1.) -> set:
    """Chunks a trace that follows the GT sheet reads: at every step the box around the front (the GT row ``t``
    steps ahead of the seed row, across the front's columns), widened by ``pad``; optionally mapped by ``matrix``."""
    gt, valid = w['gt'], w['valid']
    iu0 = (0-spec.u_lo)/spec.res
    v0 = int(round((0-spec.v_lo)/spec.res))
    v1 = int(round(((data.front_points-1)*data.point_spacing-spec.v_lo)/spec.res))
    keys = set()
    for t in range(steps+1):
        iu = int(round(iu0+t*data.step/spec.res))
        if iu >= gt.shape[0]:
            break
        ok = valid[iu, v0:v1+1]
        if not ok.any():
            continue
        row = gt[iu, v0:v1+1][ok].astype(np.float64)
        keys |= box_keys(row.min(0), row.max(0), pad, matrix, scale_pad)
    return keys


def seeds_report(cfg, patches_dir: Path, count: int, steps: int, front: float, all_val: bool, seed_split: bool,
                 max_per_patch: int, pad: float, list_bucket: bool) -> dict:
    import time
    t0 = time.time()
    cfg = copy.deepcopy(cfg)
    cfg.data.patches_dir = [str(patches_dir)]
    seeds, spec, data = draw_seeds(cfg, count, steps, front, .95, all_val=all_val, max_per_patch=max_per_patch,
                                   seed_split=seed_split)
    matrix = affine()
    scale9 = float(np.cbrt(abs(np.linalg.det(matrix[:3, :3]))))
    per_seed, union_l2, union_9 = [], set(), set()
    for w in seeds:
        k2 = step_keys(w, spec, data, steps, pad)
        k9 = step_keys(w, spec, data, steps, pad, matrix, scale9)
        c2 = classify('l2', k2, list_bucket)
        c9 = classify('9362', k9, list_bucket)
        union_l2 |= k2
        union_9 |= k9
        extent = (np.asarray(w['reach_hi'])-np.asarray(w['reach_lo'])).round(0).tolist()
        per_seed.append(dict(patch=w['meta']['patch'][-24:], axis=w['meta']['axis'], reach_extent_xyz=extent,
                             z=[round(float(w['reach_lo'][2])), round(float(w['reach_hi'][2]))],
                             l2_chunks=len(k2), l2_missing=c2['noncached']+c2['unlisted'],
                             x9362_chunks=len(k9), x9362_missing=c9['noncached']+c9['unlisted']))
    return dict(patches=str(patches_dir), count=count, drawn=len(seeds), steps=steps, front=front,
                points=int(round(front/cfg.data.point_spacing))+1, all_val=all_val, seed_split=seed_split,
                max_per_patch=max_per_patch, pad=pad,
                seeds_fully_cached_l2=sum(s['l2_missing'] == 0 for s in per_seed),
                seeds_fully_cached_both=sum(s['l2_missing'] == 0 and s['x9362_missing'] == 0 for s in per_seed),
                union_l2=classify('l2', union_l2, list_bucket), union_9362=classify('9362', union_9, list_bucket),
                seconds=round(time.time()-t0, 1), seeds=per_seed)


def mirror_report(patches_dir: Path, pad: float, list_bucket: bool) -> dict:
    """prepare_scroll's mirror rule on every patch in ``patches_dir`` (nodes ~32 vx apart, 3^3 offsets of ``pad``)."""
    import tifffile
    matrix = affine()
    scale9 = float(np.cbrt(abs(np.linalg.det(matrix[:3, :3]))))
    k2, k9 = set(), set()
    nodes = 0
    for patch in sorted(p for p in patches_dir.iterdir() if (p/'x.tif').is_file()):
        meta = json.loads((patch/'meta.json').read_text())
        xyz = np.stack([tifffile.imread(patch/f'{c}.tif') for c in 'xyz'], -1)
        ok = np.all(xyz != -1, -1)
        step = max(1, int(32*float(meta['scale'][0])))
        points = xyz[::step, ::step][ok[::step, ::step]].astype(np.float64)
        nodes += int(ok.sum())
        mapped = points@matrix[:3, :3].T+matrix[:3, 3]
        for off in np.ndindex(3, 3, 3):
            shift = (np.array(off)-1)*pad
            k2 |= chunks.keys_for_points((points+shift)[:, ::-1])
            k9 |= chunks.keys_for_points((mapped+shift*scale9)[:, ::-1])
    return dict(patches=str(patches_dir), nodes=nodes, pad=pad, l2=classify('l2', k2, list_bucket),
                x9362=classify('9362', k9, list_bucket),
                z_rows_l2=[min(k[0] for k in k2), max(k[0] for k in k2)] if k2 else None,
                z_rows_9362=[min(k[0] for k in k9), max(k[0] for k in k9)] if k9 else None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--band-patches', type=Path, required=True, help='patches cropped to the cached band (prepare.py)')
    ap.add_argument('--slab-patches', type=Path, required=True, help='geometry-only patches of the team slab')
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--count', type=int, default=40)
    ap.add_argument('--drift', type=float, default=32.)
    ap.add_argument('--list-bucket', action='store_true')
    ap.add_argument('--v-fronts', type=float, nargs='*', default=[240., 360., 480.])
    ap.add_argument('--t-fronts', type=float, nargs='*', default=[240., 360., 480., 840.])
    ap.add_argument('--v-mpp', type=int, nargs='*', default=[1, 2])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_cfg()
    pad = view_radius(cfg.model)+cfg.data.step+4+args.drift
    report = dict(view_radius=view_radius(cfg.model), read_pad=pad, villa_commit=env.VILLA_COMMIT, protocols={})
    for front in args.v_fronts:
        for mpp in args.v_mpp:
            key = f'V_valcells_front{int(front)}_mpp{mpp}'
            report['protocols'][key] = seeds_report(cfg, args.band_patches, args.count, args.steps, front, False, True,
                                                    mpp, pad, args.list_bucket)
            r = report['protocols'][key]
            print(f"{key}: drawn {r['drawn']}/{args.count}, fully cached L2 {r['seeds_fully_cached_l2']}, "
                  f"L2 and 9.362 {r['seeds_fully_cached_both']} ({r['seconds']} s)", flush=True)
    for front in args.t_fronts:
        key = f'T_slab_allval_front{int(front)}'
        report['protocols'][key] = seeds_report(cfg, args.slab_patches, args.count, args.steps, front, True, False, 0,
                                                pad, args.list_bucket)
        r = report['protocols'][key]
        print(f"{key}: drawn {r['drawn']}/{args.count}, fully cached L2 {r['seeds_fully_cached_l2']}; union L2 "
              f"{r['union_l2']}, union 9.362 {r['union_9362']} ({r['seconds']} s)", flush=True)
    report['T_slab_mirror'] = mirror_report(args.slab_patches, 96., args.list_bucket)
    print(f"T_slab_mirror: {report['T_slab_mirror']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
