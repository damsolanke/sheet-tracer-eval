import json

import numpy as np
import pytest

from synthetic import pooled_by_hand, random_traces, seed_meta, trace, write_run
from tracereval import analyze


def test_pooled_sums_over_traces_then_averages_over_steps():
    traces = [trace(ok=[10, 10, 0], kept=[8, 4, 0], cover=[6, 4, 0], real_switch=2),
              trace(ok=[10, 0, 0], kept=[10, 0, 0], cover=[9, 0, 0])]
    p = analyze.pooled(traces)
    # Step 3 has no scored and no kept points, so it drops out of both means.
    assert p['coverage'] == pytest.approx((15/20+4/10)/2)
    assert p['precision'] == pytest.approx((15/18+4/4)/2)
    assert p['real_switches'] == 1 and p['n'] == 2


def test_pooled_agrees_with_an_independent_implementation():
    traces = random_traces(30, 12, seed=1)
    p = analyze.pooled(traces)
    coverage, precision = pooled_by_hand(traces)
    assert p['coverage'] == pytest.approx(coverage, abs=1e-12)
    assert p['precision'] == pytest.approx(precision, abs=1e-12)


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    traces = random_traces(40, 10, seed=2)
    a, b = analyze.bootstrap(traces, n_boot=300), analyze.bootstrap(traces, n_boot=300)
    assert a == b
    p = analyze.pooled(traces)
    for key in ('coverage', 'precision'):
        lo, hi = a[key]
        assert lo < p[key] < hi


def test_bootstrap_of_identical_traces_has_no_width():
    traces = [trace(ok=[10, 10], kept=[6, 5], cover=[5, 4])]*20
    ci = analyze.bootstrap(traces, n_boot=200)
    p = analyze.pooled(traces)
    assert ci['coverage'] == pytest.approx([p['coverage']]*2)
    assert ci['precision'] == pytest.approx([p['precision']]*2)


def test_summary_checks_the_counts_against_the_report(tmp_path):
    run = write_run(tmp_path/'run', random_traces(8, 6, seed=3))
    s = analyze.summary(run)
    assert s['points'] == 7 and s['seeds'] == 8 and s['noncached_reads'] == 0
    report = json.loads((run/'report.json').read_text())
    report['models']['tracer7']['summary']['coverage_mean'] += 1e-6
    (run/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='do not reproduce'):
        analyze.summary(run)


def _flat_run(folder, front, cover, kept, seeds=40, noncached=0):
    """``seeds`` identical one-step traces with 10 scored points: coverage cover/10, precision cover/kept."""
    traces = [trace(ok=[10], kept=[kept], cover=[cover])]*seeds
    return write_run(folder, traces, front=front, seeds=[seed_meta(i) for i in range(seeds)], noncached=noncached)


def test_gate_applies_the_preregistered_margins(tmp_path):
    team = [(.457, .980), (.605, .980), (.548, .957)]
    runs = [_flat_run(tmp_path/'a', 240, 4, 4),   # coverage 0.40 >= 0.337, precision 1.00 >= 0.940
            _flat_run(tmp_path/'b', 480, 5, 5),   # coverage 0.50 >= 0.485
            _flat_run(tmp_path/'c', 840, 5, 5)]   # coverage 0.50 >= 0.428
    result = analyze.gate(team, runs)
    assert result['decision'] == 'GO' and result['reasons'] == []
    assert [r['coverage_floor'] for r in result['rows']] == [.337, .485, .428]
    assert [r['precision_floor'] for r in result['rows']] == [.94, .94, .917]

    runs[1] = _flat_run(tmp_path/'b2', 480, 4, 4)   # coverage 0.40 < 0.485
    runs[2] = _flat_run(tmp_path/'c2', 840, 5, 6)   # precision 0.833 < 0.917
    result = analyze.gate(team, runs)
    assert result['decision'] == 'NO-GO'
    assert result['reasons'] == ['9 points: coverage 0.400 below 0.485', '15 points: precision 0.833 below 0.917']


def test_gate_fails_integrity_on_missing_ct_or_too_few_seeds(tmp_path):
    team = [(.457, .980)]
    for run in (_flat_run(tmp_path/'a', 240, 4, 4, noncached=1), _flat_run(tmp_path/'b', 240, 4, 4, seeds=39)):
        result = analyze.gate(team, [run])
        assert result['decision'] == 'NO-GO' and result['reasons'][0].startswith('5 points: integrity')


def test_paired_matches_seeds_by_patch_orientation_and_anchor(tmp_path):
    ref_traces = random_traces(6, 8, seed=4)
    run_traces = random_traces(6, 8, seed=5)
    # The second run traced seeds 4, 1, 0 and 5 in another order, from another patch directory, with anchors that
    # differ only past the sixth decimal; seeds 2 and 3 are missing.
    order = [4, 1, 0, 5]
    run_seeds = []
    for i in order:
        s = seed_meta(i, root='/frame_c2')
        s['anchor'] = [v+4e-8 for v in s['anchor']]
        run_seeds.append(s)
    ref = write_run(tmp_path/'ref', ref_traces, seeds=[seed_meta(i, root='/frame_c0') for i in range(6)])
    run = write_run(tmp_path/'run', [run_traces[i] for i in order], seeds=run_seeds)
    p = analyze.paired(ref, run, n_boot=200)
    assert p['pairs'] == 4 and p['ref_only'] == 2
    base = analyze.pooled(ref_traces, np.array(order))
    other = analyze.pooled([run_traces[i] for i in order])
    assert p['change']['coverage'] == pytest.approx(other['coverage']-base['coverage'])
    assert p['change']['precision'] == pytest.approx(other['precision']-base['precision'])


def test_paired_change_of_a_run_against_itself_is_exactly_zero(tmp_path):
    run = write_run(tmp_path/'run', random_traces(10, 5, seed=6))
    p = analyze.paired(run, run, n_boot=100)
    assert p['change'] == {'coverage': 0., 'precision': 0.}
    assert p['ci'] == {'coverage': [0., 0.], 'precision': [0., 0.]}


def test_paired_refuses_runs_without_common_seeds(tmp_path):
    a = write_run(tmp_path/'a', random_traces(3, 4, seed=7), seeds=[seed_meta(i) for i in range(3)])
    b = write_run(tmp_path/'b', random_traces(3, 4, seed=8), seeds=[seed_meta(i+10) for i in range(3)])
    with pytest.raises(ValueError, match='no common seeds'):
        analyze.paired(a, b, n_boot=10)
