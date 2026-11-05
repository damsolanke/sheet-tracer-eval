import numpy as np
import pytest

from tracereval import pieces as P


class Cfg:
    class trace:
        spline_samples = 6


def fake_trace(lengths):
    """A stand-in for villa's trace: piece k returns lengths[k]+1 rows; position x = full-front column, y = piece id,
    z = step; confidence = 0.5 + 0.1 * piece id."""
    calls = []

    def trace(model, cfg, seed, direction, steps, device, **kw):
        k = len(calls)
        calls.append((seed[0, 0], len(seed), kw))
        width = (len(seed)-1)*6+1
        T = lengths[k]+1
        rows = np.zeros((T, width, 3), np.float32)
        rows[..., 0] = seed[0, 0]*6+np.arange(width)
        rows[..., 1] = k
        rows[..., 2] = np.arange(T)[:, None]
        return rows, np.full((T, width), .5+.1*k, np.float32)
    trace.calls = calls
    return trace


def seed(n):
    s = np.zeros((n, 3), np.float32)
    s[:, 0] = np.arange(n)
    return s


def test_starts():
    assert P.starts(15, 7, 4) == [0, 4, 8]
    assert P.starts(9, 7, 4) == [0, 2]
    assert P.starts(7, 7, 4) == [0]
    assert P.starts(5, 7, 4) == [0]
    assert P.starts(16, 7, 4) == [0, 4, 8, 9]


def test_short_seed_is_unchanged():
    t = fake_trace([60])
    rows, confs = P.piece_trace(t)(None, Cfg, seed(7), seed(7), 60, 'cpu', keep_threshold=.3)
    assert rows.shape == (61, 37, 3) and t.calls == [(0., 7, dict(keep_threshold=.3))]


def test_merge_takes_the_most_central_live_piece():
    t = fake_trace([60, 60, 60])
    rows, confs = P.piece_trace(t)(None, Cfg, seed(15), seed(15), 60, 'cpu')
    assert rows.shape == (61, 85, 3)
    assert np.array_equal(rows[0, :, 0], np.arange(85))  # every column at its own place
    owner = rows[5, :, 1]
    # piece 0 covers columns 0-36 (centre 18), piece 1 24-60 (centre 42), piece 2 48-84 (centre 66)
    assert owner[0] == 0 and owner[30] == 0 and owner[31] == 1 and owner[54] == 1 and owner[55] == 2 and owner[84] == 2
    assert [c[1] for c in t.calls] == [7, 7, 7]


def test_stopped_piece_hands_over_or_pads():
    t = fake_trace([60, 10, 60])
    rows, confs = P.piece_trace(t)(None, Cfg, seed(15), seed(15), 60, 'cpu')
    assert rows.shape == (61, 85, 3)
    # column 33 lies in pieces 0 (centre 18) and 1 (centre 42): piece 1 holds it while it traces, then piece 0
    assert rows[5, 33, 1] == 1 and rows[20, 33, 1] == 0 and np.isclose(confs[20, 33], .5)
    # column 42 lies in piece 1 only: after piece 1 stops it keeps its last position with confidence 0
    assert rows[20, 42, 1] == 1 and rows[20, 42, 2] == 10 and confs[20, 42] == 0
    t = fake_trace([10, 10, 10])
    rows, confs = P.piece_trace(t)(None, Cfg, seed(15), seed(15), 60, 'cpu')
    assert rows.shape == (11, 85, 3)  # as long as the longest piece; the benchmark pads the rest


def test_columns_no_piece_covers_get_zero_confidence():
    outs = [(np.ones((3, 37, 3)), np.ones((3, 37))), (np.ones((3, 37, 3)), np.ones((3, 37))),
            (np.ones((2, 37, 3)), np.ones((2, 37)))]
    rows, confs = P.merge(outs, [0, 24, 48], 85)
    assert confs[2, 61:].max() == 0 and confs[2, :61].min() == 1
    with pytest.raises(ValueError):
        P.merge([(np.ones((2, 37, 3)), np.ones((2, 37)))], [0], 85)
