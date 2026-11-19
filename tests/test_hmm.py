import numpy as np

from pathsnap.hmm import viterbi_segments


def test_viterbi_prefers_connected_path_over_greedy():
    # Three steps, two candidates each. Greedy picks 1,0,1; transitions only allow staying on one track.
    emis = [np.log([0.4, 0.6]), np.log([0.6, 0.4]), np.log([0.4, 0.6])]
    stay = np.log(np.array([[0.99, 1e-6], [1e-6, 0.99]]))
    assert list(viterbi_segments([0, 1, 2], emis, lambda k: stay)) == [1, 1, 1]


def test_viterbi_restarts_on_disconnect():
    emis = [np.log([0.9, 0.1]), np.log([0.1, 0.9])]
    blocked = np.full((2, 2), -np.inf)
    assert list(viterbi_segments([0, 1], emis, lambda k: blocked)) == [0, 1]
    assert list(viterbi_segments([0, 1], emis, lambda k: None)) == [0, 1]


def _backward_jitter_case(**kw):
    # A 40 m two-way road. The second fix sits 3 m behind the first on the same arc (7.5% of its length).
    # Its candidates are the same arc and the reverse arc, equally likely to the classifier.
    from pathsnap.hmm import decode_trip
    return decode_trip(
        obs=np.array([0, 1, 1]), arc=np.array([0, 0, 1]), t_along=np.array([0.5, 0.425, 0.575]),
        prob=np.array([0.9, 0.5, 0.5]), px=np.array([20.0, 17.0]), py=np.zeros(2), t=np.array([0.0, 1.0]),
        nx=np.array([0.0, 40.0]), ny=np.zeros(2), arcs=np.array([[0, 1], [1, 0]]), beta=5.0, **kw)


def test_relative_tolerance_turns_short_backward_jitter_into_a_uturn():
    assert _backward_jitter_case() == {0: 0, 1: 1}


def test_absolute_tolerance_keeps_short_backward_jitter_on_the_same_arc():
    assert _backward_jitter_case(back_tol_m=15.0) == {0: 0, 1: 0}
