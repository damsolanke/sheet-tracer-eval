"""Residual shifts of the October affine (2.399 um level 2 -> 9.362 um) on blocks around held-out mesh nodes.

Same measurement as ``windeval/fit_transform.py`` (``match`` under a fixed matrix, no refit): a 48^3 block of the
9.362 um scan on the integer grid around the mapped centre, the level-2 scan resampled onto that grid (trilinear)
through the inverse affine, phase correlation for the residual shift (target voxels) and the NCC after the integer
shift; blocks with NCC under 0.3 or a shift of 16 vx or more are dropped. Reads cached chunks only (raw 128^3 uint8
zarr v2 chunks); a block touching a chunk that the bucket has but the disk lacks is skipped and counted.

  python -m tracereval.affine_check --patches DIR --z 4402 6234 --blocks 300 --out check.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from tracereval import env

env.setup(threads=1)

import tifffile  # noqa: E402
from scipy import ndimage  # noqa: E402

from tracereval import chunks  # noqa: E402
from tracereval import feasibility as F  # noqa: E402

SIZE = 48
CH = 128


class Reader:
    """Integer-box reads from a local raw zarr level; missing chunks are zeros if the bucket lacks them too."""

    def __init__(self, level_dir: Path):
        self.dir = Path(os.path.realpath(level_dir))
        self.shape = np.array(json.loads((self.dir/'.zarray').read_text())['shape'])
        self.cache = {}

    def chunk(self, key):
        if key not in self.cache:
            path = self.dir/str(key[0])/str(key[1])/str(key[2])
            if path.is_file():
                self.cache[key] = np.fromfile(path, np.uint8).reshape(CH, CH, CH)
            else:
                kind = chunks.classify(self.dir, key)
                if kind != 'absent':
                    raise LookupError(kind)
                self.cache[key] = None
        return self.cache[key]

    def box(self, lo, hi) -> np.ndarray:
        """Voxels [lo, hi) (zyx integers) as float32."""
        lo, hi = np.asarray(lo, int), np.asarray(hi, int)
        out = np.zeros(hi-lo, np.float32)
        for cz in range(lo[0]//CH, (hi[0]-1)//CH+1):
            for cy in range(lo[1]//CH, (hi[1]-1)//CH+1):
                for cx in range(lo[2]//CH, (hi[2]-1)//CH+1):
                    c0 = np.array([cz, cy, cx])*CH
                    a, b = np.maximum(lo, c0), np.minimum(hi, c0+CH)
                    if np.any(b <= a):
                        continue
                    data = self.chunk((cz, cy, cx))
                    if data is not None:
                        out[tuple(slice(x-l, y-l) for x, y, l in zip(a, b, lo))] = \
                            data[tuple(slice(x-c, y-c) for x, y, c in zip(a, b, c0))]
        return out


def phase_shift(fixed, moving):
    """windeval.fit_transform.phase_shift, unchanged: shift s (zyx) with moving(x) ~ fixed(x + s), NCC, peak."""
    n = fixed.shape[0]
    win = np.hanning(n)
    w = win[:, None, None]*win[None, :, None]*win[None, None, :]
    fa = np.fft.fftn((fixed-fixed.mean())*w)
    fb = np.fft.fftn((moving-moving.mean())*w)
    cross = fa*np.conj(fb)
    corr = np.real(np.fft.ifftn(cross/(np.abs(cross)+1e-9)))
    peak = np.array(np.unravel_index(np.argmax(corr), corr.shape))
    shift = peak.astype(np.float64)
    for axis in range(3):
        lo, hi = peak.copy(), peak.copy()
        lo[axis] = (peak[axis]-1) % n
        hi[axis] = (peak[axis]+1) % n
        c0, cm, cp = corr[tuple(peak)], corr[tuple(lo)], corr[tuple(hi)]
        denom = cm-2*c0+cp
        if denom < 0:
            shift[axis] += .5*(cm-cp)/denom
    shift = np.where(shift > n/2, shift-n, shift)
    s = np.round(shift).astype(int)
    a = np.roll(fixed, tuple(-s), axis=(0, 1, 2))
    core = slice(abs(s).max()+1, n-abs(s).max()-1)
    aa, bb = a[core, core, core].ravel(), moving[core, core, core].ravel()
    ncc = float(np.corrcoef(aa, bb)[0, 1]) if aa.size > 10 and aa.std() > 0 and bb.std() > 0 else 0.
    return shift, ncc, float(corr.max())


def centres(patches: Path, zlo: float, zhi: float, count: int, seed: int) -> np.ndarray:
    """fit_transform's draw: up to 64 valid nodes per patch in the z range, then ``count`` of them (L2 xyz)."""
    rng = np.random.default_rng(seed)
    out = []
    for patch in sorted(p for p in patches.iterdir() if (p/'z.tif').is_file()):
        x, y, z = (tifffile.imread(patch/f'{c}.tif').astype(np.float64) for c in 'xyz')
        ok = (z > 0) & np.isfinite(z) & (z >= zlo) & (z < zhi)
        pts = np.stack([x[ok], y[ok], z[ok]], -1)
        if len(pts):
            out.append(pts[rng.choice(len(pts), min(len(pts), 64), replace=False)])
    pts = np.concatenate(out)
    return pts[rng.choice(len(pts), min(len(pts), count), replace=False)]


def measure(matrix: np.ndarray, pts: np.ndarray, src: Reader, dst: Reader) -> tuple[list, dict]:
    m = matrix[:3]
    inverse = np.linalg.inv(matrix)[:3]
    half = SIZE//2
    off = np.arange(-half, SIZE-half, dtype=np.float64)
    zz, yy, xx = np.meshgrid(off, off, off, indexing='ij')
    grid0 = np.stack([xx, yy, zz], -1).reshape(-1, 3)
    rows, skipped = [], dict(noncached=0, flat=0)
    for c_src in pts:
        c_dst = np.round(c_src@m[:, :3].T+m[:, 3])
        lo = (c_dst[::-1]-half).astype(int)
        try:
            fixed = dst.box(lo, lo+SIZE)
            grid = (grid0+c_dst)@inverse[:, :3].T+inverse[:, 3]  # target grid in source xyz
            zyx = grid[:, ::-1]
            blo = np.floor(zyx.min(0)).astype(int)-1
            bhi = np.ceil(zyx.max(0)).astype(int)+2
            block = src.box(blo, bhi)
        except LookupError:
            skipped['noncached'] += 1
            continue
        moving = ndimage.map_coordinates(block, (zyx-blo).T, order=1, mode='nearest').reshape(SIZE, SIZE, SIZE)
        if fixed.std() < 1 or moving.std() < 1:
            skipped['flat'] += 1
            continue
        shift_zyx, ncc, peak = phase_shift(fixed, moving.astype(np.float32))
        rows.append(dict(source_xyz=c_src.tolist(), target_xyz=c_dst.tolist(), shift_xyz=shift_zyx[::-1].tolist(),
                         ncc=ncc, peak=peak))
    return rows, skipped


def stats(rows: list, ncc_min: float = .3) -> dict:
    good = [r for r in rows if r['ncc'] >= ncc_min and np.abs(r['shift_xyz']).max() < SIZE/3]
    if not good:
        return dict(blocks=0)
    s = np.array([r['shift_xyz'] for r in good])
    mag = np.linalg.norm(s, axis=1)
    return dict(blocks=len(good), median=float(np.median(mag)), p90=float(np.percentile(mag, 90)),
                mean=float(mag.mean()), mean_shift_xyz=s.mean(0).round(3).tolist(),
                median_shift_xyz=np.median(s, 0).round(3).tolist(),
                ncc_median=float(np.median([r['ncc'] for r in good])))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--patches', type=Path, required=True)
    ap.add_argument('--z', type=float, nargs=2, required=True)
    ap.add_argument('--bands', type=int, default=5, help='equal z bands for the breakdown')
    ap.add_argument('--blocks', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    matrix = F.affine()
    src, dst = Reader(F.VOLUMES['l2']['local']), Reader(F.VOLUMES['9362']['local'])
    pts = centres(args.patches, *args.z, args.blocks, args.seed)
    rows, skipped = measure(matrix, pts, src, dst)
    edges = np.linspace(args.z[0], args.z[1], args.bands+1)
    bands = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if a <= r['source_xyz'][2] < b]
        bands.append(dict(z=[round(float(a), 1), round(float(b), 1)], **stats(sel)))
    out = dict(affine=str(F.AFFINE), patches=str(args.patches), z=args.z, centres=len(pts), skipped=skipped,
               overall=stats(rows), bands=bands, criterion='median <= 1 and p90 <= 2 target voxels', rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
