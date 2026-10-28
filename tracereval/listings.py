"""Bucket row listings (keys only) for the local mirrors, in the file format the integrity check reads.

``chunks.classify`` decides whether a chunk that is not on disk is ``absent`` (not in the bucket either),
``noncached`` (in the bucket, not in the mirror) or ``unlisted`` (row never listed) from the mirror's
``.windeval_present/<cz>.txt``: one ``cy/cx`` per line, written by windeval's chunk store when it listed a z row.
Rows that windeval never touched have no such file, so a run reading there would count every non-disk chunk as
``unlisted``. This module lists z rows from the bucket (S3 ListObjectsV2, keys only, no chunk data, as
``feasibility.list_row`` and windeval's ``vc`` shim do) and writes the files that are missing. An existing file is
compared with the fresh listing and never overwritten; any difference is reported (and the exit code is 1).

  python -m tracereval.listings --rows 20 62 --mirror DIR --remote PHerc0139/volumes/X.zarr/2 --out report.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'


def list_row(remote: str, cz: int) -> set:
    """Chunk keys 'cy/cx' present in bucket row ``cz`` of the array at ``remote``."""
    prefix = f'{remote}/{cz}/'
    present, token = set(), None
    while True:
        url = f'{BUCKET}/?list-type=2&prefix={urllib.parse.quote(prefix)}'
        if token:
            url += '&continuation-token='+urllib.parse.quote(token)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=120) as response:
                    text = response.read().decode()
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        present.update(key[len(prefix):] for key in re.findall(r'<Key>([^<]*)</Key>', text))
        match = re.search(r'<NextContinuationToken>([^<]*)<', text)
        if not match:
            break
        token = match.group(1)
    return present


def refresh(mirror: Path, remote: str, rows, cache: Path | None = None) -> dict:
    """List ``rows`` of ``remote``; write missing ``.windeval_present`` files of ``mirror``; compare existing ones."""
    root = Path(os.path.realpath(mirror))
    present_dir = root/'.windeval_present'
    present_dir.mkdir(exist_ok=True)
    out = dict(mirror=str(root), remote=remote, rows=[], written=[], identical=[], differ=[], cache_identical=[],
               cache_differ=[], listed_at=time.strftime('%Y-%m-%d %H:%M:%S %Z'))
    for cz in rows:
        keys = list_row(remote, cz)
        on_disk = set()
        row_dir = root/str(cz)
        if row_dir.is_dir():
            for cy in os.listdir(row_dir):
                if cy.isdigit():
                    on_disk.update(f'{cy}/{cx}' for cx in os.listdir(row_dir/cy) if cx.isdigit())
        entry = dict(cz=cz, bucket_keys=len(keys), on_disk=len(on_disk), on_disk_not_in_bucket=len(on_disk-keys),
                     in_bucket_not_on_disk=len(keys-on_disk))
        path = present_dir/f'{cz}.txt'
        if path.is_file():
            old = set(path.read_text().split())
            if old == keys:
                out['identical'].append(cz)
            else:
                out['differ'].append(dict(cz=cz, only_old=sorted(old-keys)[:20], only_new=sorted(keys-old)[:20],
                                          n_only_old=len(old-keys), n_only_new=len(keys-old)))
        else:
            tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
            tmp.write_text('\n'.join(sorted(keys)))
            tmp.replace(path)
            out['written'].append(cz)
        if cache is not None and (cache/f'{cz}.txt').is_file():
            (out['cache_identical'] if set((cache/f'{cz}.txt').read_text().split()) == keys
             else out['cache_differ']).append(cz)
        out['rows'].append(entry)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mirror', type=Path, required=True, help='local zarr level directory (symlinks are followed)')
    ap.add_argument('--remote', required=True, help='bucket path of the same level, e.g. PHerc0139/volumes/X.zarr/2')
    ap.add_argument('--rows', type=int, nargs=2, required=True, metavar=('FIRST', 'LAST'), help='z chunk rows, inclusive')
    ap.add_argument('--cache', type=Path, help='earlier listings (catalog/listings/...) to compare with')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    report = refresh(args.mirror, args.remote, range(args.rows[0], args.rows[1]+1), args.cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    worst = [r for r in report['rows'] if r['on_disk_not_in_bucket']]
    print(json.dumps(dict(remote=args.remote, rows=args.rows, written=len(report['written']),
                          identical=report['identical'], differ=[d['cz'] for d in report['differ']],
                          cache_identical=len(report['cache_identical']), cache_differ=report['cache_differ'],
                          bucket_keys=sum(r['bucket_keys'] for r in report['rows']),
                          on_disk=sum(r['on_disk'] for r in report['rows']),
                          in_bucket_not_on_disk=sum(r['in_bucket_not_on_disk'] for r in report['rows']),
                          rows_with_disk_keys_not_in_bucket=[r['cz'] for r in worst])), flush=True)
    raise SystemExit(1 if report['differ'] or worst else 0)


if __name__ == '__main__':
    main()
