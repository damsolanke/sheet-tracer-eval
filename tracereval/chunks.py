"""Which CT chunks are cached locally, which exist in the bucket, and which ones a run tried to read.

The October mirrors (``~/kaggle-data/vesuvius/zarr/...``) are partial: a chunk is on disk only if the winding-model
runs touched it. villa's ``ChunkedArray`` reads a chunk file that is not on disk as fill (zeros), which is also what
it does for chunks the bucket leaves out (masked regions outside the scroll). ``instrument()`` records every chunk a
reader could not find on disk and classifies it with the mirror's bucket listing (``.windeval_present/<cz>.txt``,
written by windeval when it listed a z row of chunks):

  absent     the bucket has no such chunk either: zeros are correct
  noncached  the bucket has it, the mirror does not: the run saw zeros where there is CT (invalid input)
  unlisted   the row was never listed: unknown, counted as invalid
"""
from __future__ import annotations

import os
import threading
from collections import defaultdict
from pathlib import Path

import numpy as np

CHUNK_BYTES = 128**3  # raw uint8 chunks of the open-data OME-Zarr volumes

_LOCK = threading.Lock()
MISSES: dict = defaultdict(lambda: defaultdict(set))  # array dir -> class -> {key}
_LISTINGS: dict = {}


def listing(array_dir: Path, cz: int):
    """Set of 'cy/cx' present in the bucket for z chunk row ``cz``, or None if the row was never listed."""
    array_dir = Path(os.path.realpath(array_dir))
    key = (str(array_dir), cz)
    if key not in _LISTINGS:
        path = array_dir/'.windeval_present'/f'{cz}.txt'
        _LISTINGS[key] = set(path.read_text().split()) if path.is_file() else None
    return _LISTINGS[key]


def classify(array_dir: Path, key) -> str:
    cz, cy, cx = (int(v) for v in key)
    if (Path(array_dir)/str(cz)/str(cy)/str(cx)).is_file():
        return 'cached'
    present = listing(array_dir, cz)
    if present is None:
        return 'unlisted'
    return 'noncached' if f'{cy}/{cx}' in present else 'absent'


def cached_keys(array_dir: Path) -> set:
    """Chunk keys (cz, cy, cx) on disk in a '/'-separated zarr level directory."""
    root = Path(os.path.realpath(array_dir))
    keys = set()
    for cz in os.listdir(root):
        if not cz.isdigit():
            continue
        for cy in os.listdir(root/cz):
            if not cy.isdigit():
                continue
            for cx in os.listdir(root/cz/cy):
                if cx.isdigit():
                    keys.add((int(cz), int(cy), int(cx)))
    return keys


def instrument() -> None:
    """Record the chunks villa's ChunkedArray could not find on disk (the values it returns are unchanged)."""
    from vesuvius.neural_tracing.tracers.shared.crop_sampling import volume as V
    if getattr(V.ChunkedArray, '_tracereval_instrumented', False):
        return
    original = V.ChunkedArray._load

    def _load(self, key):
        result = original(self, key)
        if result is None:
            kind = classify(Path(self.path), key)
            with _LOCK:
                MISSES[os.path.realpath(self.path)][kind].add(tuple(int(v) for v in key))
        return result

    V.ChunkedArray._load = _load
    V.ChunkedArray._tracereval_instrumented = True


def snapshot() -> dict:
    """Counts of distinct missing chunk keys per class, over all arrays."""
    with _LOCK:
        out = defaultdict(int)
        for kinds in MISSES.values():
            for kind, keys in kinds.items():
                out[kind] += len(keys)
        return dict(out)


def keys_snapshot() -> dict:
    """The missing chunk keys themselves: array dir -> class -> sorted [cz, cy, cx] lists."""
    with _LOCK:
        return {array: {kind: sorted(list(k) for k in keys) for kind, keys in kinds.items()}
                for array, kinds in MISSES.items()}


def keys_for_points(points_zyx: np.ndarray, chunks=(128, 128, 128)) -> set:
    """Chunk keys holding level voxels ``points_zyx`` (N, 3) floats."""
    if not len(points_zyx):
        return set()
    k = np.unique(np.floor(np.asarray(points_zyx)/np.asarray(chunks)).astype(np.int64), axis=0)
    return set(map(tuple, k.tolist()))


def keys_for_box(lo_zyx, hi_zyx, chunks=(128, 128, 128)) -> set:
    """Chunk keys overlapping the voxel box [lo, hi] (zyx, inclusive)."""
    c0 = np.floor(np.asarray(lo_zyx)/np.asarray(chunks)).astype(int)
    c1 = np.floor(np.asarray(hi_zyx)/np.asarray(chunks)).astype(int)
    return {(z, y, x) for z in range(c0[0], c1[0]+1) for y in range(c0[1], c1[1]+1) for x in range(c0[2], c1[2]+1)}
