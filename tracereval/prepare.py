"""villa's ``scripts/prepare_scroll.py`` without downloads: patches from local meshes, CT occupancy from cached chunks.

Same inputs and rules as prepare_scroll (segments registered on ``--volume-id`` from the bucket catalogue, coordinates
times ``--coord-scale``, nodes in empty CT plus a ``--margin`` along the surface set invalid, held-out z band and
training band ``--gap`` away, ``--min-nodes``), with three differences:

* meshes are read from the local copies under ``$VESUVIUS_DATA/data/<path>`` (same files as the bucket's);
* the occupancy test reads only chunks already on disk. A chunk the bucket lacks is empty, as in prepare_scroll; a
  chunk the bucket has but the mirror does not ("noncached"), or whose row was never listed, counts as empty too, so
  its nodes and the margin around them are masked. Patches therefore cover only cached CT;
* nothing is mirrored or downloaded; ``--parts`` picks which patch sets to write.

  python -m tracereval.prepare --scroll PHerc0139 --volume-id 20260102150214 --coord-scale 0.25 \\
      --level-dir ~/kaggle-data/vesuvius/tracereval/volumes/PHerc0139_2399um_l2.zarr/0 --heldout-z 3904 4288 \\
      --parts heldout --out ~/kaggle-data/vesuvius/tracereval/prepared/PHerc0139_2399l2_smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tracereval import env

env.setup()

import tifffile  # noqa: E402
from scipy import ndimage  # noqa: E402

from tracereval import chunks  # noqa: E402
from vesuvius.neural_tracing.tracers.sheets.scripts import prepare_scroll as PS  # noqa: E402


class CachedOccupancy(PS.Occupancy):
    """prepare_scroll's Occupancy over the chunks on disk only; records why a chunk had no cells."""

    def __init__(self, level_dir: Path, zarray: dict):
        self.dir, self.zarray = Path(level_dir), zarray
        self.shape, self.chunk = np.array(zarray['shape']), np.array(zarray['chunks'])
        self.cells, self.kinds = {}, {}
        self.absent = set()
        self.codec = None
        if zarray.get('compressor'):
            import numcodecs
            self.codec = numcodecs.get_codec(zarray['compressor'])

    def load(self, keys) -> int:
        for key in keys:
            if key in self.cells:
                continue
            kind = chunks.classify(self.dir, key)
            self.kinds[key] = kind
            path = self.dir/PS.chunk_name(self.zarray, key)
            self.cells[key] = self._cells(path.read_bytes()) if kind == 'cached' else None
        return 0


def read_segment_local(path: str, coord_scale: float):
    """prepare_scroll.read_segment on the local copy of a bucket tifxyz."""
    root = env.VESUVIUS_DATA/'data'/path
    meta = json.loads((root/'meta.json').read_text())
    xyz = np.stack([tifffile.imread(root/f'{c}.tif').astype(np.float32) for c in 'xyz'], -1)
    xyz = np.where(xyz == -1, -1, xyz*coord_scale).astype(np.float32)
    meta['scale'] = [s/coord_scale for s in meta['scale']]
    return xyz, meta


def default_band(meta: dict, segs: list, volume_id: str, coord_scale: float, shape_z: float, fraction: float):
    """prepare_scroll's default held-out z band: the middle ``fraction`` of the segments' z extent."""
    boxes = [np.array(g['properties']['volume_coverage'][volume_id]['bbox_transformed'])*coord_scale
             for g in segs if volume_id in (g['properties'].get('volume_coverage') or {})]
    z0, z1 = min(b[0][2] for b in boxes), max(b[1][2] for b in boxes)
    z0, z1 = max(z0, 0.), min(z1, float(shape_z))
    mid, half = (z0+z1)/2, fraction*(z1-z0)/2
    return mid-half, mid+half


def registered_segments(catalog: dict, scroll: str, volume_id: str) -> list:
    meta = catalog['samples'][scroll]
    return sorted((g for g in meta['segments'].values() if PS.segment_path(g, 'tifxyz-transformed', volume_id)),
                  key=lambda g: g['long_id'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--catalog', default=str(env.CATALOG))
    ap.add_argument('--scroll', required=True)
    ap.add_argument('--volume-id', required=True)
    ap.add_argument('--coord-scale', type=float, required=True)
    ap.add_argument('--level-dir', required=True, help='local level directory read as level 0 (cached chunks)')
    ap.add_argument('--margin', type=float, default=32.)
    ap.add_argument('--heldout-z', type=float, nargs=2)
    ap.add_argument('--heldout-fraction', type=float, default=.2)
    ap.add_argument('--gap', type=float, default=256.)
    ap.add_argument('--min-nodes', type=int, default=20000)
    ap.add_argument('--parts', choices=('heldout', 'train', 'both'), default='heldout')
    ap.add_argument('--segments', nargs='*', help='only these long_ids (default: all registered)')
    ap.add_argument('--no-occupancy', action='store_true',
                    help='geometry only: skip the empty-CT test (for estimating data needs before any CT is local)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    catalog = json.loads(Path(args.catalog).read_text())
    segs = registered_segments(catalog, args.scroll, args.volume_id)
    level_dir = Path(args.level_dir).expanduser()
    zarray = json.loads((level_dir/'.zarray').read_text())
    if args.heldout_z:
        lo, hi = args.heldout_z
    else:
        lo, hi = default_band(catalog['samples'][args.scroll], segs, args.volume_id, args.coord_scale,
                              zarray['shape'][0], args.heldout_fraction)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    occupancy = CachedOccupancy(level_dir, zarray)
    print(f'{len(segs)} registered segments; held-out z [{lo:.1f}, {hi:.1f}], training z outside '
          f'[{lo-args.gap:.1f}, {hi+args.gap:.1f}]', flush=True)
    parts = {'heldout': ('heldout_patches',), 'train': ('train_patches',),
             'both': ('train_patches', 'heldout_patches')}[args.parts]
    summary = []
    t0 = time.time()
    for g in segs:
        name = g['long_id']
        if args.segments and name not in args.segments:
            continue
        path = PS.segment_path(g, 'tifxyz-transformed', args.volume_id)
        xyz, smeta = read_segment_local(path, args.coord_scale)
        valid = np.all(xyz != -1, -1) & np.all(np.isfinite(xyz), -1)
        z = xyz[..., 2]
        # Held-out patches only: nodes more than the margin (plus 4 vx) away in z from the band cannot be within the
        # margin of a band node along the surface, so only nodes near the band get the occupancy test.
        reach = valid & (((z >= lo-args.margin-4) & (z <= hi+args.margin+4)) if parts == ('heldout_patches',)
                         else valid)
        void = np.zeros(valid.shape, bool)
        if not args.no_occupancy:
            void[reach] = occupancy.empty(xyz[reach])
        cell = 1./float(smeta['scale'][0])
        near = ndimage.distance_transform_edt(~void, sampling=cell) < args.margin if void.any() else void
        kept = valid & ~near
        row = dict(segment=name, valid=int(valid.sum()), tested=int(reach.sum()), empty_or_uncached=int(void[reach].sum()),
                   kept=int(kept.sum()))
        for part in parts:
            sel = kept & (((z < lo-args.gap) | (z > hi+args.gap)) if part == 'train_patches' else ((z >= lo) & (z <= hi)))
            row[part] = int(sel.sum())
            if row[part] < args.min_nodes:
                continue
            rows, cols = np.flatnonzero(sel.any(1)), np.flatnonzero(sel.any(0))
            r, c = slice(rows[0], rows[-1]+1), slice(cols[0], cols[-1]+1)
            part_xyz = np.where(sel[r, c, None], xyz[r, c], -1).astype(np.float32)
            PS.write_segment(out/part/name, part_xyz, smeta)
        summary.append(row)
        print(json.dumps(row), flush=True)
    kinds = {}
    for kind in occupancy.kinds.values():
        kinds[kind] = kinds.get(kind, 0)+1
    report = dict(vars(args), heldout_z=[lo, hi], segments=summary, occupancy_chunks=kinds,
                  seconds=round(time.time()-t0, 1), villa_commit=env.VILLA_COMMIT)
    (out/'prepare_cached.json').write_text(json.dumps(report, indent=1))
    print(f'occupancy chunks by kind {kinds}; {time.time()-t0:.0f} s -> {out}', flush=True)


if __name__ == '__main__':
    main()
