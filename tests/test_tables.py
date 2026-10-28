import json
import math
from pathlib import Path

import pytest

from synthetic import random_traces, seed_meta, trace, write_run
from tracereval import analyze, c0_table, c123_table

RESULTS = Path(__file__).resolve().parents[1]/'results'


def test_traced_coverage_leaves_rows_after_a_stop_out_of_the_denominator():
    # One trace runs all 3 steps, the other stops after step 1; its later rows are scored (ok) but not seen.
    traces = [trace(ok=[10, 10, 10], kept=[9, 8, 8], cover=[9, 8, 6]),
              trace(ok=[10, 10, 10], kept=[10, 0, 0], cover=[7, 0, 0], seen=[10, 0, 0], stopped=2)]
    assert analyze.pooled(traces)['coverage'] == pytest.approx((16/20+8/20+6/20)/3)
    assert c0_table.traced_coverage(traces) == pytest.approx((16/20+8/10+6/10)/3)


def test_c0_row_counts_stops_and_full_length_traces(tmp_path):
    steps = 12
    traces = [trace(ok=[8]*steps, kept=[6]*steps, cover=[5]*steps),
              trace(ok=[8]*steps, kept=[6]*3+[0]*9, cover=[5]*3+[0]*9, seen=[8]*3+[0]*9, stopped=4),
              trace(ok=[8]*steps, kept=[6]*11+[0], cover=[5]*11+[0], seen=[8]*11+[0], stopped=12)]
    row = c0_table.row(write_run(tmp_path/'run', traces))
    assert row['full_length'] == 1 and row['stopped_before_10'] == 1
    assert row['steps_traced_median'] == 11 and row['steps_traced_mean'] == pytest.approx((12+3+11)/3)


def _grid(root, conds, fronts, thresholds, seeds=6, steps=8):
    """Synthetic runs named as the tables expect; every condition and threshold traces the same seeds."""
    k = 0
    for cond in conds:
        for front in fronts:
            for t in thresholds:
                k += 1
                write_run(root/f'A0_{cond}_f{front}_t{t:g}', random_traces(seeds, steps, seed=k), front=float(front),
                          seeds=[seed_meta(i, root=f'/{cond}') for i in range(seeds)])


def test_c0_table_builds_every_row_and_paired_threshold_change(tmp_path):
    _grid(tmp_path, ['C0'], c0_table.FRONTS, c0_table.THRESHOLDS)
    c0_table.main(['--runs-dir', str(tmp_path), '--out', str(tmp_path/'t.json')])
    table = json.loads((tmp_path/'t.json').read_text())
    assert [(r['points'], r['threshold']) for r in table['rows']] == \
        [(p, t) for p in (5, 7, 9, 15) for t in (.3, .4, .5)]
    assert len(table['paired_thresholds']) == 8
    for p in table['paired_thresholds']:
        assert p['against'] == .4 and p['pairs'] == 6
        assert p['change']['coverage'] == pytest.approx(p['run_metrics']['coverage']-p['ref_metrics']['coverage'])


def test_c123_reading_follows_the_plan():
    def entry(cov_hi, prec, prec_hi):
        return dict(ci=dict(coverage=[-1., cov_hi], precision=[-1., prec_hi]), change=dict(precision=prec))
    resample, none = 'resample', 'no measurable effect at this sample size'
    assert c123_table.reading(entry(-.001, 0., .01)) == resample      # coverage interval below zero
    assert c123_table.reading(entry(.001, -.01, -.001)) == resample   # precision -0.01 or less, interval below zero
    assert c123_table.reading(entry(.001, -.009, -.001)) == none      # precision drop too small
    assert c123_table.reading(entry(.001, -.02, .001)) == none        # precision interval reaches zero
    assert c123_table.reading(entry(0., 0., 0.)) == none


def test_c123_table_reads_only_the_resampling_comparisons(tmp_path):
    _grid(tmp_path, ['C0', 'C1', 'C2', 'C3'], c123_table.FRONTS, c123_table.THRESHOLDS)
    c123_table.main(['--runs-dir', str(tmp_path), '--out', str(tmp_path/'t.json')])
    table = json.loads((tmp_path/'t.json').read_text())
    assert table['label'].startswith('exploratory')
    assert len(table['runs']) == 16 and len(table['paired']) == 20
    for p in table['paired']:
        assert ('reading' in p) == (p['ref_cond'] == 'C3' and p['cond'] in ('C1', 'C2'))
        assert p['pairs'] == 6


def _same_numbers(a, b, path=''):
    """Every number and string in ``a`` equals ``b``, except the run paths (they name a different folder)."""
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for key in a:
            if key not in ('run', 'ref'):
                _same_numbers(a[key], b[key], f'{path}.{key}')
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _same_numbers(x, y, f'{path}[{i}]')
    elif isinstance(a, float) and math.isnan(a):
        assert isinstance(b, float) and math.isnan(b), path
    else:
        assert a == b, path


@pytest.mark.skipif(not (RESULTS/'A0_C0_table.json').is_file(), reason='results/ not present')
def test_shipped_counts_rebuild_the_shipped_tables(tmp_path):
    """The per-trace counts in results/ reproduce every number in the three tables the write-up quotes."""
    c0_table.main(['--runs-dir', str(RESULTS), '--out', str(tmp_path/'c0.json')])
    c123_table.main(['--runs-dir', str(RESULTS), '--out', str(tmp_path/'c123.json')])
    gate = analyze.gate([(.457, .980), (.605, .980), (.548, .957)],
                        [RESULTS/f'A0_C0_f{f}_t0.4' for f in (240, 480, 840)])
    for name, rebuilt in (('A0_C0_table.json', json.loads((tmp_path/'c0.json').read_text())),
                          ('A0_C123_table.json', json.loads((tmp_path/'c123.json').read_text())),
                          ('A0_C0_gate.json', json.loads(json.dumps(gate)))):
        _same_numbers(json.loads((RESULTS/name).read_text()), rebuilt, name)
    assert gate['decision'] == 'NO-GO'
