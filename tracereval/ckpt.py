"""The sheet tracer checkpoint, loaded like ``sheets.checkpoint.load_checkpoint`` but without unpickling code.

villa's loader calls ``torch.load(..., weights_only=False)``. The public ``last.pt`` holds only tensors, numbers,
strings and dicts, so ``weights_only=True`` reads the same objects; model, config and weights are then built by
villa's own ``config.from_dict``, ``model.build_model`` and ``model.load_model_state``, as in its loader.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path, device, raw: bool = False):
    """Model (EMA weights if the run kept them, unless ``raw``), config and checkpoint dict, as villa's loader."""
    import torch
    from vesuvius.neural_tracing.tracers.sheets import config as C
    from vesuvius.neural_tracing.tracers.sheets.model import build_model, load_model_state

    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    cfg = C.from_dict(checkpoint['config'], strict=False)
    model = build_model(cfg).to(device)
    load_model_state(model, checkpoint['model'] if raw or not checkpoint.get('ema') else checkpoint['ema'])
    return model, cfg, checkpoint


def install() -> None:
    """Route villa's ``load_checkpoint`` (as imported by the benchmark and inference modules) through this one."""
    from vesuvius.neural_tracing.tracers.sheets import checkpoint as villa_checkpoint
    from vesuvius.neural_tracing.tracers.sheets.evaluation import benchmark
    from vesuvius.neural_tracing.tracers.sheets.tracing import inference
    for module in (villa_checkpoint, benchmark, inference):
        module.load_checkpoint = load_checkpoint
