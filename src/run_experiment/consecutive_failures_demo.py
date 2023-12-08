import numpy as np
from typing import Tuple, List, Any, Iterable
from collections import defaultdict
import logging
import datetime as dt
import pickle as pkl
from pathlib import Path

"""
TODO
- test asymmetric demos
- visualize data

"""

EPS = np.finfo(float).eps


def make_transition_matrices(p_switch: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make transition matrices for the symmetric and asymmetric cases.
    """

    transitions_symmetric = np.ones((2, 2)) * p_switch
    for i in range(2):
        transitions_symmetric[i, i] = 1 - p_switch

    transitions_asymmetric = np.ones((2, 2)) * p_switch
    transitions_asymmetric[0, 0] = 1 - p_switch
    transitions_asymmetric[0, 1] = 0
    transitions_asymmetric[1, 1] = 1
    return transitions_symmetric, transitions_asymmetric


def run_update(prior: np.ndarray, p_omission: np.ndarray, transitions: np.ndarray) -> np.ndarray:
    """
    Run a single update step for the HMM.
    """
    joint_prob = prior * p_omission
    joint_prob = joint_prob / (joint_prob.sum() + EPS)
    posterior = np.dot(transitions, joint_prob)
    return posterior  # / (posterior.sum() + EPS)


def run_symmetric_demo(prior: np.ndarray, p_active_reward: float, transitions: np.ndarray, n_trials=10) -> np.ndarray:
    """
    Run the demo task for n_trials steps. Return the probability of staying on each trial as an n_trials x 2 array.
    Col 1 is p(stay) and col 2 is p(switch).
    """

    priors = []
    p_omission = np.array([1 - p_active_reward, p_active_reward])
    for _ in range(n_trials):
        prior = run_update(prior, p_omission, transitions)
        priors.append(prior.copy())
    return np.array(priors)


def run_asymmetric_demo(prior: np.ndarray, p_active_reward: float, asymmetric_transitions: np.ndarray,
                        n_trials=10) -> np.ndarray:
    priors = []
    p_omission = np.array([1 - p_active_reward, 1])
    for _ in range(n_trials):
        prior = run_update(prior, p_omission, asymmetric_transitions)
        priors.append(prior.copy())
    return np.array(priors)


def demo_consecutive_failures(p_switch: float, p_active_reward: float, n_trials=10)\
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate running the task at different reward probabilities and switch probabilities.
    """

    sym_mat, asym_mat = make_transition_matrices(p_switch)
    prior = np.array([1-p_switch, p_switch])
    sym_priors = run_symmetric_demo(prior, p_active_reward, sym_mat, n_trials=n_trials)
    symmetric_pstay = sym_priors[:, 0]

    asym_priors = run_asymmetric_demo(prior, p_active_reward, asym_mat, n_trials=n_trials)
    asymmetric_pstay = asym_priors[:, 0]
    return symmetric_pstay, asymmetric_pstay


def get_trial_switch_ix(pstay: Iterable) -> int:
    return next((i + 1 for i, x in enumerate(pstay) if x <= .5), np.nan)   # add 1 to get the trial number from zero-index


def get_trials_to_switch(pstay: np.ndarray):
    """
    Use collected runs to determine the number of trials/consecutive failures to switch for a given p_active_reward.

    """
    get_trial = lambda run: next((i for i, x in enumerate(pstay) if x <= .5), None)
    if len(pstay.shape) == 2:
        return np.apply_along_axis(get_trial_switch_ix, 1, pstay)
    else:
        return get_trial_switch_ix(pstay)


def save_runs(data) -> None:
    date = str(dt.date.today().isoformat())
    save_dir = Path('../../data/processed/{}/demo_runs'.format(date))
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    p = save_dir / 'inference_consecutive_failures.pkl'
    with open(p, 'wb') as f:
        pkl.dump(data, f)
    logging.info("[***] Data saved in {}".format(p.name))


def main():
    p_switch = .2
    # p_active_reward = [.9, .8, .7, .6, .5]
    p_active_reward = np.arange(start=.1, stop=1, step=.1)

    symmetric_runs = []
    asymmetric_runs = []
    for p in p_active_reward:
        sym_stay, asym_pstay = demo_consecutive_failures(p_switch, p, n_trials=10)
        symmetric_runs.append(sym_stay)
        asymmetric_runs.append(asym_pstay)

    symmetric_runs = np.array(symmetric_runs)
    asymmetric_runs = np.array(asymmetric_runs)
    sym_switch = get_trials_to_switch(symmetric_runs)
    asym_switch = get_trials_to_switch(asymmetric_runs)

    data = {'symmetric': symmetric_runs, 'asymmetric': asymmetric_runs, 'sym_switch': sym_switch,
            'asym_switch': asym_switch, 'p_active_reward': p_active_reward, 'p_switch': p_switch}
    save_runs(data)


if __name__ == '__main__':
    main()
