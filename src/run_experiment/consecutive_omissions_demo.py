import numpy as np
from typing import Tuple, List, Any, Iterable, Callable
import datetime as dt
import pickle as pkl
from pathlib import Path


EPS = np.finfo(float).eps


def make_symmetric_transition_matrix(p_switch: float) -> np.ndarray:
    transitions_symmetric = np.ones((2, 2)) * p_switch
    for i in range(2):
        transitions_symmetric[i, i] = 1 - p_switch
    return transitions_symmetric


def make_asymmetric_transition_matrix(p_switch: float) -> np.ndarray:
    transitions_asymmetric = np.ones((2, 2)) * p_switch
    transitions_asymmetric[0, 0] = 1 - p_switch
    transitions_asymmetric[0, 1] = 0
    transitions_asymmetric[1, 1] = 1
    return transitions_asymmetric


def update_matrix_prior(prior: np.ndarray, p_omission: np.ndarray, transitions: np.ndarray) -> np.ndarray:
    """
    Run a single update step for the HMM.
    """
    joint_prob = prior * p_omission
    joint_prob = joint_prob / (joint_prob.sum() + EPS)
    posterior = np.dot(transitions, joint_prob)
    return posterior  # / (posterior.sum() + EPS)


def update_recursion_prior(prior: float, p_omission: float, p_switch: float) -> float:
    """
    Update the prior for the active side in the asymmetric case.
    Uses a simplified equation.
    """
    posterior = (1-p_switch) * p_omission * prior / (1 + prior * (p_omission-1))
    # posterior = p_omission * prior / (1 + prior * (p_omission-1)) # testing for exclusion of p_switch
    return posterior


def run_recursion_task(p_active_reward: float, p_switch: float, n_trials=10) -> np.ndarray:
    p_omission = 1 - p_active_reward
    _prior = 1 - p_switch
    priors = []
    for _ in range(n_trials):
        _prior = update_recursion_prior(_prior, p_omission, p_switch)
        priors.append(_prior)
    return np.array(priors)


def run_matrix_task(p_switch: float, p_omission: np.ndarray, transitions: np.ndarray, n_trials=10) -> np.ndarray:
    """
    Run the demo task for n_trials steps. Return the probability of staying on each trial as an n_trials x 2 array.
    Col 1 is p(stay) and col 2 is p(switch).
    """
    _prior = np.array([1 - p_switch, p_switch])
    priors = []
    for _ in range(n_trials):
        _prior = update_matrix_prior(_prior, p_omission, transitions)
        priors.append(_prior.copy())
    return np.array(priors)


def run_symmetric_demo(p_active_reward: float, p_switch: float, n_trials=10) -> np.ndarray:
    """
    Run the demo task for n_trials steps. Return the probability of staying on each trial as an n_trials x 2 array.
    Col 1 is p(stay) and col 2 is p(switch).
    """
    sym_mat = make_symmetric_transition_matrix(p_switch)
    p_omission = np.array([1 - p_active_reward, p_active_reward])
    sym_priors = run_matrix_task(p_switch=p_switch, p_omission=p_omission, transitions=sym_mat, n_trials=n_trials)
    symmetric_pstay = sym_priors[:, 0]
    return symmetric_pstay


def run_asymmetric_demo(p_active_reward: float, p_switch: float, n_trials=10) -> np.ndarray:
    """
    Run the demo task for n_trials steps. Return the probability of staying on each trial as an n_trials x 2 array.
    Col 1 is p(stay) and col 2 is p(switch).
    """
    asym_mat = make_asymmetric_transition_matrix(p_switch)
    p_omission = np.array([1 - p_active_reward, 1])
    asym_priors = run_matrix_task(p_switch=p_switch, p_omission=p_omission, transitions=asym_mat,
                                  n_trials=n_trials)
    asymmetric_pstay = asym_priors[:, 0]
    return asymmetric_pstay


def get_trial_switch_ix(p_stay: Iterable) -> int:
    return next((i + 1 for i, x in enumerate(p_stay) if x <= .5), np.nan)   # add 1 to get the trial number from zero-index


def get_trials_to_switch(p_stay: np.ndarray):
    """
    Use collected runs to determine the number of trials/consecutive failures to switch for a given p_active_reward.
    """
    if len(p_stay.shape) == 1:
        return float(get_trial_switch_ix(p_stay))
    elif len(p_stay.shape) == 2:
        return np.asarray([get_trial_switch_ix(row) for row in p_stay], dtype=float)
    elif len(p_stay.shape) == 3:
        return np.asarray(
            [[get_trial_switch_ix(row) for row in plane] for plane in p_stay],
            dtype=float,
        )
    else:
        raise ValueError("p_stay must be 1, 2, or 3 dimensional; got {} dimensions.".format(len(p_stay.shape)))


def save_runs(data: dict, fname: str, note: str='') -> None:
    date = str(dt.date.today().isoformat())
    save_dir = Path('../../data/processed/{}/consecutive_omissions'.format(date))
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    p = save_dir / (fname + '.pkl')
    with open(p, 'wb') as f:
        pkl.dump(data, f)
    print("[***] Data saved in {}".format(p.name))

    if note:
        with open(save_dir / 'readme.txt', 'w') as f:
            f.write(note)


def loop_params(p_switch: np.ndarray, p_active_reward: np.ndarray, runtypes: List[Callable], n_trials=10) -> List[Any]:
    """
    Loop over the parameters and run the task for each combination.
    """
    all_results = [[] for _ in range(len(runtypes))]
    for q in p_switch:
        temp_results = [[] for _ in range(len(runtypes))]
        for p in p_active_reward:
            tmp = [run(p_active_reward=p, p_switch=q, n_trials=n_trials) for run in runtypes]
            for i, t in enumerate(tmp):
                temp_results[i].append(t.copy())

        for i, t in enumerate(temp_results):
            all_results[i].append(np.array(t))

    all_results = [np.stack(r, axis=0) for r in all_results]
    return all_results


def run_qlearning_demo(decay: np.ndarray, n_trials=10) -> np.ndarray:
    tmp = []
    for d in decay:
        tmp.append(np.array([d ** i for i in range(n_trials)]))
    relative_value = np.array(tmp)
    return relative_value


def main():
    p_switch = np.arange(start=.1, stop=1, step=.1)
    p_active_reward = np.arange(start=.1, stop=1, step=.1)

    runtypes = [run_symmetric_demo, run_asymmetric_demo, run_recursion_task]
    all_results = loop_params(p_switch, p_active_reward, runtypes, n_trials=10)

    symmetric_runs, asymmetric_runs, recursion_runs = all_results
    sym_switch, asym_switch, rec_switch = [get_trials_to_switch(r) for r in (symmetric_runs, asymmetric_runs, recursion_runs)]

    data = {'symmetric': symmetric_runs, 'asymmetric': asymmetric_runs, 'recursion': recursion_runs,
            'sym_switch': sym_switch, 'asym_switch': asym_switch, 'rec_switch': rec_switch,
            'p_active_reward': p_active_reward, 'p_switch': p_switch}

    decay = np.arange(start=.1, stop=1, step=.1)
    qlearning_run = run_qlearning_demo(decay, n_trials=10)
    RL_switch = get_trials_to_switch(qlearning_run)
    data['decay'] = decay
    data['qlearning'] = qlearning_run
    data['RL_switch'] = RL_switch

    note = ('INFERENCE: \n'
            'symmetric and asymmetric matrices represent the probability of staying at a chosen port/side after a\n'
            'reward omission. Matrices are N x R x T, where N is the number of p_switch values, R is the number of\n'
            'p_active_reward values, and T is the number of trials. Trials to switch collapses these matrices along \n'
            'the T dimension, so that the output is N x R.\n\n'
            'RL: \n'
            'qlearning matrix represents the probability of staying at a chosen port/side after a\n'
            'reward omission. Matrix is N x T, where N is the number of decay values and T is the number of trials.\n'
            'Trials to switch collapses these matrices along the T dimension, so that the output is N x 1.')

    # save_runs(data, 'consecutive_omissions', note=note)


if __name__ == '__main__':
    main()
