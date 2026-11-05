import json
import math
from pathlib import Path

from tracereval import pieces_table

RESULTS = Path(__file__).resolve().parents[1]/'results'


def _same(a, b, path='table'):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _same(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f'{path}[{i}]')
    elif isinstance(a, float):
        assert math.isclose(a, b, rel_tol=0, abs_tol=1e-12), path
    else:
        assert a == b, path


def test_long_row_table_rebuilds_from_the_per_trace_counts():
    stored = json.loads((RESULTS/'long_rows'/'A0_pieces_table.json').read_text())
    _same(pieces_table.build(RESULTS), stored)


def test_decision_numbers_quoted_in_the_readme():
    t = json.loads((RESULTS/'long_rows'/'A0_pieces_table.json').read_text())
    d = t['decision_15']
    assert t['identity_7'] is True
    assert (d['pooled']['n'], d['released']['n'], d['fresh']['n']) == (77, 40, 37)
    assert round(d['pooled']['change']['coverage'], 3) == 0.179
    assert [round(x, 3) for x in d['pooled']['ci']['coverage']] == [0.124, 0.234]
    assert round(d['released']['stock']['coverage'], 3) == 0.403 and round(d['released']['pieces']['coverage'], 3) == 0.626
    assert round(d['fresh']['stock']['coverage'], 3) == 0.408 and round(d['fresh']['pieces']['coverage'], 3) == 0.538
    assert d['pooled']['ci']['coverage'][0] > 0 and d['pooled']['change']['precision'] >= -0.010
    assert d['pooled']['pieces']['real_switches'] <= d['pooled']['stock']['real_switches']+0.10*d['pooled']['n']
