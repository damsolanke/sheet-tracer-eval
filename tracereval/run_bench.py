"""Run villa's trace benchmark (``evaluation/benchmark.py``) unchanged, with bookkeeping around it.

Everything after ``--`` goes to ``benchmark.main`` as is. Around it this wrapper:

* imports villa from the pinned source snapshot (``tracereval.env``) and limits threads;
* loads the checkpoint with ``weights_only=True`` (``tracereval.ckpt``; same model, config and EMA weights);
* records every CT chunk the run could not find on disk, per trace (``tracereval.chunks``);
* optionally resamples the volume (``--voxel-um``, ``tracereval.resample``) so one model voxel spans another
  physical size;
* optionally replaces the per-seed figures (they show CT) with nothing (``--no-figures``);
* optionally traces a seed row longer than N points as overlapping N-point pieces (``--piece-points N``,
  ``--piece-stride S``; ``tracereval.pieces``), each piece with villa's own ``trace``;
* writes ``run_meta.json`` next to ``report.json``: arguments, commits, checkpoint hash, package versions, device,
  per-trace wall time and chunk misses, peak memory.

  python -m tracereval.run_bench --threads 1 --device cpu --no-figures -- --checkpoints CKPT --names tracer \\
      --patches-dir DIR --all-val --seeds 1 --steps 3 --front 360 --threshold 0.4 --volume VOL --out OUT
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--' in argv:
        split = argv.index('--')
        ours, theirs = argv[:split], argv[split+1:]
    else:
        ours, theirs = argv, []
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--device', default='cpu', help='cpu or mps; passed to the benchmark as --device')
    ap.add_argument('--no-figures', action='store_true')
    ap.add_argument('--resample', help='JSON of tracereval.resample.ResampleSpec (affine + voxel size)')
    ap.add_argument('--piece-points', type=int, help='trace rows longer than this as overlapping pieces (off by default)')
    ap.add_argument('--piece-stride', type=int, default=2, help='with --piece-points: a piece starts every S points')
    args = ap.parse_args(ours)

    from tracereval import env
    env.setup(threads=args.threads)
    import numpy as np
    import torch
    from tracereval import chunks, ckpt
    from vesuvius.neural_tracing.tracers.sheets.evaluation import benchmark

    chunks.instrument()
    ckpt.install()
    if args.resample:
        from tracereval import resample
        resample.install(json.loads(Path(args.resample).read_text()))
    if args.no_figures:
        benchmark.seed_figure = lambda *a, **k: None

    if args.piece_points:
        from tracereval import pieces
        benchmark.trace = pieces.piece_trace(benchmark.trace, args.piece_points, args.piece_stride)

    traces = []
    original_trace = benchmark.trace

    def timed_trace(*a, **k):
        before = chunks.snapshot()
        t0 = time.time()
        rows, confs = original_trace(*a, **k)
        after = chunks.snapshot()
        traces.append(dict(seconds=round(time.time()-t0, 3), steps_traced=int(len(rows)-1),
                           new_missing_chunks={key: after.get(key, 0)-before.get(key, 0) for key in after}))
        print(f'[tracereval] trace {len(traces)}: {traces[-1]}', flush=True)
        return rows, confs

    benchmark.trace = timed_trace

    # Per-trace counts behind the pooled summary (benchmark.score's own outputs), for bootstrap intervals over seeds.
    scored = []
    original_score = benchmark.score

    def recorded_score(*a, **k):
        sc = original_score(*a, **k)
        keep = ('ok', 'same', 'seen', 'kept', 'cover', 'off', 'caught')
        scored.append(dict({key: np.asarray(sc[key]).astype(int).tolist() for key in keep},
                           first_switch=sc['first_switch'], first_real_switch=sc['first_real_switch'],
                           off_kept_points=sc['off_kept_points'], lost=sc['lost'], stopped=sc['stopped'],
                           dist_median=[float(np.median(d)) if len(d) else None for d in sc['dist']]))
        return sc

    benchmark.score = recorded_score
    if '--device' not in theirs:
        theirs += ['--device', args.device]
    out = Path(theirs[theirs.index('--out')+1]) if '--out' in theirs else None
    t0 = time.time()
    benchmark.main(theirs)
    wall = time.time()-t0
    meta = dict(wrapper_args=vars(args), benchmark_argv=theirs, villa_commit=env.VILLA_COMMIT,
                villa_src=str(env.VILLA_SRC), checkpoint=str(env.CHECKPOINT), checkpoint_sha256=env.CHECKPOINT_SHA256,
                hf_revision=env.HF_REVISION, python=platform.python_version(), torch=torch.__version__,
                numpy=np.__version__, platform=platform.platform(), wall_seconds=round(wall, 1), traces=traces,
                missing_chunks_total=chunks.snapshot(), missing_chunk_keys=chunks.keys_snapshot(),
                peak_rss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30, 2))
    if out is not None:
        (out/'run_meta.json').write_text(json.dumps(meta, indent=1))
        (out/'per_trace.json').write_text(json.dumps(scored))
    print(f'[tracereval] wall {wall:.1f} s, missing chunks {meta["missing_chunks_total"]}, '
          f'peak RSS {meta["peak_rss_gib"]} GiB', flush=True)


if __name__ == '__main__':
    main()
