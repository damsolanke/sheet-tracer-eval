"""Exact chunk lists a later download step would fetch (keys present in the bucket and missing from the local mirror).

Three options, recomputed with the same rules as ``feasibility*.py`` and written as JSON (no CT is read or fetched):

  a1_first40   the benchmark's first 40 A1 seeds (PHerc0139 validation cells in the cached band, 5-point fronts,
               60 steps, reads plus 32 vx drift), both scans
  a0_slab      prepare_scroll's mirror of PHerc0139's default held-out slab (96 vx around every held-out node), both scans
  pherc0343    96 vx around Nieuwlaar's w047 / w048 meshes on PHerc0343's 8.64 um scan (real 8.64 um check)
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from tracereval import env

env.setup(threads=1)

from tracereval import chunks  # noqa: E402
from tracereval import feasibility as F  # noqa: E402
from vesuvius.neural_tracing.tracers.sheets.evaluation.benchmark import draw_seeds  # noqa: E402
from vesuvius.neural_tracing.tracers.sheets.sampling import view_radius  # noqa: E402

BUCKET = F.BUCKET
P343_LOCAL = env.VESUVIUS_DATA/'zarr'/'PHerc0343'/'volumes'/'20250521140437-8.640um-1.2m-116keV-masked.zarr'/'0'
P343_REMOTE = 'PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr/0'


def missing(local: Path, remote: str, keys: set) -> list:
    out, rows = [], {}
    for key in sorted(keys):
        kind = chunks.classify(local, key)
        if kind == 'unlisted':
            if key[0] not in rows:
                rows[key[0]] = F.list_row(remote, key[0])
            kind = 'noncached' if f'{key[1]}/{key[2]}' in rows[key[0]] else 'absent'
        if kind == 'noncached':
            out.append(list(key))
    return out


def entry(local: Path, remote: str, keys: set) -> dict:
    need = missing(local, remote, keys)
    return dict(url=f'{BUCKET}/{remote}', local_mirror=str(local), chunks_considered=len(keys), to_fetch=len(need),
                gib=round(len(need)*chunks.CHUNK_BYTES/2**30, 3), keys_zyx=need)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--band-patches', type=Path, required=True)
    ap.add_argument('--slab-patches', type=Path, required=True)
    ap.add_argument('--p343-meshes', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = F.load_cfg()
    pad = view_radius(cfg.model)+cfg.data.step+4+32.
    matrix = F.affine()
    scale9 = float(np.cbrt(abs(np.linalg.det(matrix[:3, :3]))))
    l2, x9 = F.VOLUMES['l2'], F.VOLUMES['9362']

    c = copy.deepcopy(cfg)
    c.data.patches_dir = [str(args.band_patches)]
    seeds, spec, data = draw_seeds(c, 40, 60, 240., .95, all_val=False, max_per_patch=0, seed_split=True)
    k2, k9 = set(), set()
    for w in seeds:
        k2 |= F.step_keys(w, spec, data, 60, pad)
        k9 |= F.step_keys(w, spec, data, 60, pad, matrix, scale9)
    a1 = dict(option='a1_first40', seeds=[w['meta'] for w in seeds], read_pad=pad,
              l2=entry(l2['local'], l2['remote'], k2), x9362=entry(x9['local'], x9['remote'], k9))
    (args.out/'a1_first40.json').write_text(json.dumps(a1))
    print('a1_first40', a1['l2']['to_fetch'], a1['l2']['gib'], a1['x9362']['to_fetch'], a1['x9362']['gib'], flush=True)

    import tifffile
    k2, k9 = set(), set()
    for patch in sorted(p for p in args.slab_patches.iterdir() if (p/'x.tif').is_file()):
        meta = json.loads((patch/'meta.json').read_text())
        xyz = np.stack([tifffile.imread(patch/f'{ch}.tif') for ch in 'xyz'], -1)
        ok = np.all(xyz != -1, -1)
        step = max(1, int(32*float(meta['scale'][0])))
        points = xyz[::step, ::step][ok[::step, ::step]].astype(np.float64)
        mapped = points@matrix[:3, :3].T+matrix[:3, 3]
        for off in np.ndindex(3, 3, 3):
            shift = (np.array(off)-1)*96.
            k2 |= chunks.keys_for_points((points+shift)[:, ::-1])
            k9 |= chunks.keys_for_points((mapped+shift*scale9)[:, ::-1])
    a0 = dict(option='a0_slab', heldout_z_level2=[4402.4, 6233.6], l2=entry(l2['local'], l2['remote'], k2),
              x9362=entry(x9['local'], x9['remote'], k9))
    (args.out/'a0_slab.json').write_text(json.dumps(a0))
    print('a0_slab', a0['l2']['to_fetch'], a0['l2']['gib'], a0['x9362']['to_fetch'], a0['x9362']['gib'], flush=True)

    keys = set()
    for patch in sorted(p for p in args.p343_meshes.iterdir() if (p/'x.tif').is_file()):
        meta = json.loads((patch/'meta.json').read_text())
        xyz = np.stack([tifffile.imread(patch/f'{ch}.tif') for ch in 'xyz'], -1).astype(np.float64)
        ok = np.all(xyz != -1, -1) & np.all(np.isfinite(xyz), -1)
        step = max(1, int(32*float(meta['scale'][0])))
        points = xyz[::step, ::step][ok[::step, ::step]]
        for off in np.ndindex(3, 3, 3):
            keys |= chunks.keys_for_points((points+(np.array(off)-1)*96.)[:, ::-1])
    p3 = dict(option='pherc0343', meshes='w047_R5B2, w048_R5B2 (github.com/Nieuwlaar/pherc343-first-letters, outputs/)',
              volume=entry(P343_LOCAL, P343_REMOTE, keys))
    (args.out/'pherc0343.json').write_text(json.dumps(p3))
    print('pherc0343', p3['volume']['to_fetch'], p3['volume']['gib'], flush=True)


if __name__ == '__main__':
    main()
