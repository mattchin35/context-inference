from src.run_experiment import consecutive_failures_demo
from src.visualization import plot_inference_consecutive_failures
import numpy as np
import numpy.testing as npt


def test_make_transition_matrices():
    p_switch = .1
    sym_mat, asym_mat = consecutive_failures_demo.make_transition_matrices(p_switch)

    npt.assert_array_equal(sym_mat, np.array([[.9,.1], [.1,.9]]))
    npt.assert_array_equal(asym_mat, np.array([[.9, 0], [.1, 1]]))


def test_run_symmetric_update():
    p_switch = .1
    p_active_reward = .9

    sym_mat, asym_mat = consecutive_failures_demo.make_transition_matrices(p_switch)
    posterior = np.array([1 - p_switch, p_switch])
    p_omission = np.array([1 - p_active_reward, p_active_reward])
    for _ in range(1000):
        posterior = consecutive_failures_demo.run_update(posterior, p_omission, sym_mat)
        npt.assert_allclose(np.sum(posterior), 1, atol=1e-6)


def test_run_symmetric_demo():
    p_switch = .1
    p_active_reward = .9

    sym_mat, _ = consecutive_failures_demo.make_transition_matrices(p_switch)
    posterior = np.array([1 - p_switch, p_switch])
    pstay = consecutive_failures_demo.run_symmetric_demo(posterior, p_active_reward, sym_mat, n_trials=10)
    assert type(pstay) is np.ndarray
    assert pstay.shape == (10, 2)

    # p(stay) should monotonically decrease
    assert np.all(pstay[1:,0] <= pstay[:-1,0])


def test_run_asymmetric_demo():
    p_switch = .1
    p_active_reward = .9

    _, asym_mat = consecutive_failures_demo.make_transition_matrices(p_switch)
    posterior = np.array([1 - p_switch, p_switch])
    pstay = consecutive_failures_demo.run_asymmetric_demo(posterior, p_active_reward, asym_mat, n_trials=10)
    assert type(pstay) is np.ndarray
    assert pstay.shape == (10, 2)

    # p(stay) should monotonically decrease
    assert np.all(pstay[1:,0] <= pstay[:-1,0])


def test_demo_consecutive_failures():
    p_switch = .1
    p_active_reward = .9

    ntrials = 10
    sym_pstay, asym_pstay = consecutive_failures_demo.demo_consecutive_failures(p_switch, p_active_reward,
                                                                               n_trials=ntrials)
    assert type(sym_pstay) is np.ndarray
    assert type(asym_pstay) is np.ndarray
    assert sym_pstay.shape == (ntrials,)
    assert asym_pstay.shape == (ntrials,)

    assert np.all(sym_pstay[1:] <= sym_pstay[:-1])
    assert np.all(asym_pstay[1:] <= asym_pstay[:-1])


def test_trial_switch_ix():
    pstay = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, .1])
    trial_switch_ix = consecutive_failures_demo.get_trial_switch_ix(pstay)
    assert trial_switch_ix == 5

    pstay = np.array([10, 10, 10, 10, 10, 10])
    trial_switch_ix = consecutive_failures_demo.get_trial_switch_ix(pstay)
    assert np.isnan(trial_switch_ix)

    pstay = np.array([.3, .2, .1, .1, .1, .1])
    ix = consecutive_failures_demo.get_trial_switch_ix(pstay)
    assert ix == 1


def test_get_trials_to_switch():
    pstay = np.array([[.9, .8, .7, .6, .5, .4]])
    ix = consecutive_failures_demo.get_trials_to_switch(pstay)
    npt.assert_array_equal(ix, np.array([5]))

    pstay = np.array([[10, 10, 10, 10, 10, 10]])
    ix = consecutive_failures_demo.get_trials_to_switch(pstay)
    npt.assert_array_equal(ix, np.array([np.nan]))

    pstay = np.array([[.9, .8, .7, .6, .5, .4],
                     [.3, .2, .1, .1, .1, .1],
                     [10, 10, 10, 10, 10, 10]])
    ix = consecutive_failures_demo.get_trials_to_switch(pstay)
    npt.assert_array_equal(ix, np.array([5, 1, np.nan]))

