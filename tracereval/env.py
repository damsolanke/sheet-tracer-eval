"""Paths and import setup for villa's sheet tracer, run from a source snapshot without installing villa.

Environment variables:
  TRACEREVAL_DATA  data root (default ~/kaggle-data/vesuvius/tracereval): snapshots, prepared patches, runs, temp files
  VILLA_SRC        villa ``vesuvius/src`` directory to import from (default: the fe64ea3 snapshot under the data root)
  VESUVIUS_DATA    the October data root with the CT mirrors and meshes (default ~/kaggle-data/vesuvius)
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('TRACEREVAL_DATA', '~/kaggle-data/vesuvius/tracereval')).expanduser()
VESUVIUS_DATA = Path(os.environ.get('VESUVIUS_DATA', '~/kaggle-data/vesuvius')).expanduser()
VILLA_COMMIT = 'fe64ea3bbf4f4f775f202beae2b87812f2208553'
VILLA_SRC = Path(os.environ.get('VILLA_SRC', str(DATA/'src_snapshots'/'villa_fe64ea3'/'src'))).expanduser()
CHECKPOINT = VESUVIUS_DATA/'hf'/'sheet_tracer_7pt_gap60'/'last.pt'
CHECKPOINT_SHA256 = 'b1d133bcb8bf3595fb9f3676baca82ad64acae3010a55a65b1dda25b2cc853ec'
HF_REVISION = '443497f0f584f6b3e414b4379c6ad4ed1d7faacc'
CATALOG = DATA/'catalog'/'metadata_20261020_decompressed.json'

_DONE = False


def setup(threads: int | None = None) -> None:
    """Import paths, temp directory and thread limits; idempotent.

    villa's top-level ``vesuvius/__init__.py`` imports the catalogue and volume extras (zarr, fsspec, ...), which the
    sheet tracer does not use. An empty package object with the snapshot's ``vesuvius`` directory as its path lets
    ``vesuvius.neural_tracing...`` and ``vesuvius.tifxyz`` import exactly as in villa, without running that file.
    """
    global _DONE
    if threads is not None:
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS'):
            os.environ[key] = str(threads)
    if _DONE:
        return
    tmp = DATA/'tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(tmp)
    os.environ.setdefault('MPLCONFIGDIR', str(tmp/'mpl'))
    os.environ.setdefault('NUMBA_CACHE_DIR', str(tmp/'numba'))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    import tempfile
    tempfile.tempdir = str(tmp)
    if not (VILLA_SRC/'vesuvius'/'neural_tracing'/'tracers'/'sheets').is_dir():
        raise SystemExit(f'villa snapshot not found at {VILLA_SRC}')
    if 'vesuvius' not in sys.modules:
        package = types.ModuleType('vesuvius')
        package.__path__ = [str(VILLA_SRC/'vesuvius')]
        package.__file__ = str(VILLA_SRC/'vesuvius'/'__init__.py')
        sys.modules['vesuvius'] = package
    for path in (str(VILLA_SRC), str(ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    if threads is not None:
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    _DONE = True


def pick_device(requested: str = 'cpu'):
    import torch
    if requested == 'auto':
        return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    return torch.device(requested)
