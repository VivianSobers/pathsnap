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
