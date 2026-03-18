import numpy as np
import numpy.testing as npt

from src.run_experiment import consecutive_omissions_demo as demo


def test_make_transition_matrices():
    p_switch = 0.1
    sym_mat = demo.make_symmetric_transition_matrix(p_switch)
    asym_mat = demo.make_asymmetric_transition_matrix(p_switch)

    npt.assert_array_equal(sym_mat, np.array([[0.9, 0.1], [0.1, 0.9]]))
    npt.assert_array_equal(asym_mat, np.array([[0.9, 0.0], [0.1, 1.0]]))


def test_run_symmetric_update():
    p_switch = 0.1
    p_active_reward = 0.9

    sym_mat = demo.make_symmetric_transition_matrix(p_switch)
    posterior = np.array([1 - p_switch, p_switch])
    p_omission = np.array([1 - p_active_reward, p_active_reward])
    for _ in range(1000):
        posterior = demo.update_matrix_prior(posterior, p_omission, sym_mat)
        npt.assert_allclose(np.sum(posterior), 1, atol=1e-6)


def test_run_symmetric_demo():
    pstay = demo.run_symmetric_demo(p_active_reward=0.9, p_switch=0.1, n_trials=10)

    assert isinstance(pstay, np.ndarray)
    assert pstay.shape == (10,)
    assert np.all(pstay[1:] <= pstay[:-1])


def test_run_asymmetric_demo():
    pstay = demo.run_asymmetric_demo(p_active_reward=0.9, p_switch=0.1, n_trials=10)

    assert isinstance(pstay, np.ndarray)
    assert pstay.shape == (10,)
    assert np.all(pstay[1:] <= pstay[:-1])


def test_trial_switch_ix():
    pstay = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1])
    assert demo.get_trial_switch_ix(pstay) == 5

    pstay = np.array([10, 10, 10, 10, 10, 10])
    assert np.isnan(demo.get_trial_switch_ix(pstay))

    pstay = np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.1])
    assert demo.get_trial_switch_ix(pstay) == 1


def test_get_trials_to_switch():
    pstay = np.array([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]])
    npt.assert_array_equal(demo.get_trials_to_switch(pstay), np.array([5]))

    pstay = np.array([[10, 10, 10, 10, 10, 10]])
    npt.assert_array_equal(demo.get_trials_to_switch(pstay), np.array([np.nan]))

    pstay = np.array(
        [
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            [0.3, 0.2, 0.1, 0.1, 0.1, 0.1],
            [0.8] * 6,
        ]
    )
    npt.assert_array_equal(demo.get_trials_to_switch(pstay), np.array([5, 1, np.nan]))

    pstay = np.stack([pstay, pstay], axis=0)
    npt.assert_array_equal(
        demo.get_trials_to_switch(pstay),
        np.array([[5, 1, np.nan], [5, 1, np.nan]]),
    )
