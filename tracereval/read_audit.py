"""Rerun a benchmark run and list, for every trace, the CT chunks it read that are not on disk.

``run_bench`` records a missing chunk once, for the first trace that reads it: villa's ``ChunkedArray`` caches every
chunk it loads (a missing one as None) and ``tracereval.chunks`` keeps a set, so a later trace that reads the same
missing chunk is not counted. This audit wraps ``ChunkedArray.chunk``, which every read passes through, cached or not,
and runs ``tracereval.run_bench`` unchanged, so each trace's own missing reads are listed. The rerun writes
``per_trace.json`` as run_bench does; compare it with the original run's file before applying the audit to that run.

  python -m tracereval.read_audit AUDIT_JSON -- RUN_BENCH_ARGS ...
"""
from __future__ import annotations

import json
import os
import sys
import threading
from collections import defaultdict
from pathlib import Path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[1] != '--':
        raise SystemExit('usage: python -m tracereval.read_audit AUDIT_JSON -- RUN_BENCH_ARGS ...')
    out_json, bench_args = Path(argv[0]), argv[2:]
    ours = bench_args[:bench_args.index('--')] if '--' in bench_args else bench_args
    threads = int(ours[ours.index('--threads')+1]) if '--threads' in ours else 1

    from tracereval import env
    env.setup(threads=threads)  # before numpy and torch load, as run_bench does
    from tracereval import chunks, run_bench
    from vesuvius.neural_tracing.tracers.shared.crop_sampling import volume as V

    lock = threading.Lock()
    marks = [0]  # chunks.snapshot() calls so far: run_bench takes one before and one after each trace
    reads = defaultdict(set)  # mark -> {(array dir, key)} returned as missing while that mark was current
    original_chunk, original_snapshot = V.ChunkedArray.chunk, chunks.snapshot

    def chunk(self, key):
        arr = original_chunk(self, key)
        if arr is None:
            with lock:
                reads[marks[0]].add((os.path.realpath(self.path), tuple(int(v) for v in key)))
        return arr

    def snapshot():
        with lock:
            marks[0] += 1
        return original_snapshot()

    V.ChunkedArray.chunk = chunk
    chunks.snapshot = snapshot
    run_bench.main(bench_args)

    home = str(Path.home())
    traces = []
    for i in range((marks[0]-1)//2):  # trace i ran while mark 2i+1 was current
        kinds = defaultdict(list)
        for array, key in sorted(reads.get(2*i+1, ())):
            kinds[chunks.classify(Path(array), key)].append(list(key))
        traces.append({kind: len(keys) for kind, keys in kinds.items()} | {'keys': dict(kinds)})
    outside = sorted(k for mark, keys in reads.items() if mark % 2 == 0 for _, k in keys)
    arrays = sorted({a.replace(home, '~', 1) for keys in reads.values() for a, _ in keys})
    out_json.write_text(json.dumps(dict(run_bench_args=[a.replace(home, '~') for a in bench_args], arrays=arrays,
                                        traces=traces, missing_reads_outside_traces=len(outside)), indent=1))
    bad = [i for i, t in enumerate(traces) if t.get('noncached') or t.get('unlisted')]
    print(f'[read_audit] {len(traces)} traces; traces that read noncached or unlisted chunks: {bad}', flush=True)


if __name__ == '__main__':
    main()
