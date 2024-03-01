from src.run_experiment import consecutive_omissions_demo as demo
from src.visualization import plot_consecutive_omissions as plt
import numpy as np
import numpy.testing as npt


def test_make_transition_matrices():
    p_switch = .1
    sym_mat = demo.make_symmetric_transition_matrix(p_switch)
    asym_mat = demo.make_asymmetric_transition_matrix(p_switch)

    npt.assert_array_equal(sym_mat, np.array([[.9,.1], [.1,.9]]))
    npt.assert_array_equal(asym_mat, np.array([[.9, 0], [.1, 1]]))


def test_run_symmetric_update():
    p_switch = .1
    p_active_reward = .9

    sym_mat = demo.make_symmetric_transition_matrix(p_switch)
    posterior = np.array([1 - p_switch, p_switch])
    p_omission = np.array([1 - p_active_reward, p_active_reward])
    for _ in range(1000):
        posterior = demo.update_matrix_prior(posterior, p_omission, sym_mat)
        npt.assert_allclose(np.sum(posterior), 1, atol=1e-6)


def test_run_symmetric_demo():
    p_switch = .1
    p_active_reward = .9

    pstay = demo.run_symmetric_demo(p_active_reward, p_switch, n_trials=10)
    assert type(pstay) is np.ndarray
    assert pstay.shape == (10,)

    # p(stay) should monotonically decrease
    assert np.all(pstay[1:] <= pstay[:-1])


def test_run_asymmetric_demo():
    p_switch = .1
    p_active_reward = .9

    pstay = demo.run_asymmetric_demo(p_active_reward, p_switch, n_trials=10)
    assert type(pstay) is np.ndarray
    assert pstay.shape == (10,)

    # p(stay) should monotonically decrease
    assert np.all(pstay[1:] <= pstay[:-1])


def test_trial_switch_ix():
    pstay = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, .1])
    trial_switch_ix = demo.get_trial_switch_ix(pstay)
    assert trial_switch_ix == 5

    pstay = np.array([10, 10, 10, 10, 10, 10])
    trial_switch_ix = demo.get_trial_switch_ix(pstay)
    assert np.isnan(trial_switch_ix)

    pstay = np.array([.3, .2, .1, .1, .1, .1])
    ix = demo.get_trial_switch_ix(pstay)
    assert ix == 1


def test_get_trials_to_switch():
    pstay = np.array([[.9, .8, .7, .6, .5, .4]])
    ix = demo.get_trials_to_switch(pstay)
    npt.assert_array_equal(ix, np.array([5]))

    pstay = np.array([[10, 10, 10, 10, 10, 10]])
    ix = demo.get_trials_to_switch(pstay)
    npt.assert_array_equal(ix, np.array([np.nan]))

    pstay = np.array([[.9, .8, .7, .6, .5, .4],
                     [.3, .2, .1, .1, .1, .1],
                     [.8]*6])
    ix = demo.get_trials_to_switch(pstay)
    npt.assert_array_equal(ix, np.array([5, 1, np.nan]))

    pstay = np.stack([pstay, pstay], axis=0)
    ix = demo.get_trials_to_switch(pstay)
    npt.assert_array_equal(ix, np.array([[5,1, np.nan],
                                         [5,1, np.nan]]))

