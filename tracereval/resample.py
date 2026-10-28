"""Run the unchanged tracer in a model frame whose voxel is a chosen physical size, reading another scan.

Frames: ``L2`` is the 2.399 um scan at level 2 (the training protocol, where the GT patches live); ``F9`` is the
9.362 um scan at level 0, reached from L2 by a fitted affine ``A`` (``x9 = A x_l2``). A condition's model frame is
``xm = s * x9``: ``s = 1`` reads the 9.362 um scan natively, ``s = 9.362 / v`` presents it with ``v`` um voxels (8.64:
the geometry of an 8.64 um scan read natively; ``1 / 1.0283``: the training voxel, i.e. the scan resampled to about
9.6 um). GT patches are written in the model frame (``transform_patches``); step, crops and tolerances stay in model
voxels, as the checkpoint defines them.

``install(spec)`` patches three functions of villa at runtime; the numbers they return for ``s = 1`` are unchanged:

* ``tracing.inference.read_block``: for ``s != 1`` reads the 9.362 um block covering the model-frame box and returns
  it with its 9.362 um origin;
* ``shared.crop_sampling.gpu.to_block_grid``: for ``s != 1`` divides sample points by ``s`` before the usual
  level-voxel mapping, so every crop sample is one trilinear read of the 9.362 um scan (no second interpolation);
* ``evaluation.benchmark.in_split`` (held-out scoring): maps scored points back to L2 before taking split cells, so
  every condition scores the same validation cells.

The volume's ``shape`` in model voxels (the "front left the volume" stop) comes from a thin wrapper around villa's
``ChunkedArray`` that ``tracing.inference`` constructs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def load_affine(path) -> np.ndarray:
    m = np.eye(4)
    m[:3] = np.asarray(json.loads(Path(path).read_text())['matrix_source_to_target_xyz'], np.float64)
    return m


def frame_matrix(spec: dict) -> np.ndarray:
    """4x4 matrix L2 xyz -> model-frame xyz for a spec (``affine`` path or None for L2 itself, ``scale`` s)."""
    m = np.eye(4) if not spec.get('affine') else load_affine(spec['affine'])
    out = m.copy()
    out[:3] *= float(spec.get('scale', 1.))
    return out


def install(spec: dict) -> None:
    import torch
    from vesuvius.neural_tracing.tracers.shared.crop_sampling import gpu
    from vesuvius.neural_tracing.tracers.shared.crop_sampling import volume as V
    from vesuvius.neural_tracing.tracers.sheets.evaluation import benchmark
    from vesuvius.neural_tracing.tracers.sheets.tracing import inference

    s = float(spec.get('scale', 1.))
    to_l2 = np.linalg.inv(frame_matrix(spec))

    if spec.get('split_from_l2', True) and spec.get('affine'):
        original_in_split = benchmark.in_split

        def in_split(points, data, split='val'):
            m = torch.as_tensor(to_l2, dtype=points.dtype, device=points.device)
            mapped = points@m[:3, :3].T+m[:3, 3]
            return original_in_split(mapped, data, split)

        benchmark.in_split = in_split

    if s == 1.:
        return

    class ScaledArray:
        """villa's reader of the 9.362 um array (``inner``), reporting the array's shape in model voxels."""

        def __init__(self, path, cache_bytes=2 << 30):
            self.inner = V.ChunkedArray(path, cache_bytes)
            self.shape = tuple(int(math.ceil(n*s)) for n in self.inner.shape)

    def read_block(volume, pos, pad, level):
        if level != 0:
            raise ValueError('scaled frames support level-0 models only')
        lo, hi = (pos.min(0)-pad)/s, (pos.max(0)+pad)/s  # model frame -> 9.362 um voxels (xyz)
        origin = np.floor(lo[::-1]).astype(np.int64)-1
        size = np.ceil((hi-lo)[::-1]).astype(np.int64)+4
        return volume.inner.read(origin, size), origin

    original_grid = gpu.to_block_grid

    def to_block_grid(points, origin, level, size):
        return original_grid(points/s, origin, level, size)

    inference.ChunkedArray = ScaledArray
    inference.read_block = read_block
    gpu.to_block_grid = to_block_grid


def transform_patches(src: Path, dst: Path, spec: dict) -> dict:
    """Write every tifxyz patch of ``src`` into the model frame of ``spec``; meta ``scale`` follows the linear scale."""
    import tifffile
    m = frame_matrix(spec)
    linear = float(np.cbrt(abs(np.linalg.det(m[:3, :3]))))
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for patch in sorted(p for p in Path(src).iterdir() if (p/'x.tif').is_file()):
        meta = json.loads((patch/'meta.json').read_text())
        xyz = np.stack([tifffile.imread(patch/f'{c}.tif') for c in 'xyz'], -1).astype(np.float64)
        ok = np.all(xyz != -1, -1) & np.all(np.isfinite(xyz), -1)
        out = np.full(xyz.shape, -1., np.float32)
        out[ok] = (xyz[ok]@m[:3, :3].T+m[:3, 3]).astype(np.float32)
        (dst/patch.name).mkdir(exist_ok=True)
        for i, c in enumerate('xyz'):
            tifffile.imwrite(dst/patch.name/f'{c}.tif', np.ascontiguousarray(out[..., i]))
        meta['scale'] = [float(v)/linear for v in meta['scale']]
        meta['tracereval_frame'] = dict(spec, linear_scale=linear)
        (dst/patch.name/'meta.json').write_text(json.dumps(meta))
        count += 1
    return dict(patches=count, linear_scale=linear, matrix=m.tolist())


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Write patches in a model frame (see module docstring).')
    ap.add_argument('src', type=Path)
    ap.add_argument('dst', type=Path)
    ap.add_argument('--affine', help='JSON with matrix_source_to_target_xyz (L2 -> 9.362 um)')
    ap.add_argument('--scale', type=float, default=1.)
    args = ap.parse_args(argv)
    spec = dict(affine=args.affine, scale=args.scale)
    result = transform_patches(args.src, args.dst, spec)
    (args.dst/'frame.json').write_text(json.dumps(dict(spec, **result), indent=1))
    print(json.dumps(dict(spec, patches=result['patches'], linear_scale=result['linear_scale'])))


if __name__ == '__main__':
    main()
