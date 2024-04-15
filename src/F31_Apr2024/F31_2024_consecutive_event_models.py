import numpy as np
import scipy as sp
from typing import Tuple, List, Any, Iterable, Union
import datetime as dt
import pickle as pkl
from pathlib import Path
import src.consecutive_omissions.consecutive_omissions_demo as cod
import src.consecutive_omissions.inference_agent_omissions as iao
from icecream import ic
import matplotlib.pyplot as plt


RIGHT_IX = 0
LEFT_IX = 1
eps = np.finfo(float).eps


def get_action_ix(action: int) -> int:
    return action * 2 - 1


def FQL_omission_update(Q: np.ndarray, decay: float, reward: float) -> Tuple[np.ndarray, float]:
    """
    One-step omission update for a Forgetting Q-learning agent.
    Q: 1D array of Q-values for each action
    decay: decay rate for Q-values, from 0 to 1
    """

    Qnew = Q * decay
    Qnew[0] += (1 - decay) * reward
    value = Qnew[RIGHT_IX] - Qnew[LEFT_IX]  # relative value of R vs L actions
    return Qnew, value


def inference_agent_value_update(prior: Union[float, np.ndarray], p_active_reward: float) -> Union[float, np.ndarray]:
    """prior should be a scalar or a 1D array of priors for each trial."""
    expected_rew_R = p_active_reward * prior
    expected_rew_L = p_active_reward * (1-prior)
    value = expected_rew_R - expected_rew_L  # relative value of L vs R actions
    # value = sp.special.logit(prior)
    return value


def consecutive_omissions(p_switch: float, p_active_reward: float, decay: float, n_trials: int):
    # Inference
    priors = iao.run_recursion_task(p_active_reward, p_switch, n_trials)
    inference_values = inference_agent_value_update(priors, p_active_reward)

    # Reinforcement learning
    Qstart = np.array([1, 0]).astype(float)
    Q = [Qstart]
    values = [Qstart[0]-Qstart[1]]
    for i in range(n_trials):
        q, v = FQL_omission_update(Q[-1], decay, reward=0)
        Q.append(q)
        values.append(v)

    qlearning_run = np.array(values)
    return priors, inference_values, qlearning_run


def consecutive_rewards(p_switch: float, p_active_reward: float, decay: float, n_trials: int):
    # qlearning from zero
    Qstart = np.array([0,0]).astype(float)
    Q = [Qstart]
    values = [0]
    for i in range(n_trials):
        q, v = FQL_omission_update(Q[-1], decay, 1)
        Q.append(q)
        values.append(v)
    Q_zero = np.array(values)
    # ic(values)

    # qlearning from negative maximum
    Qstart = np.array([0, 1]).astype(float)
    Q = [Qstart]
    values = [Qstart[0]-Qstart[1]]
    for i in range(n_trials):
        q, v = FQL_omission_update(Q[-1], decay, 1)
        Q.append(q)
        values.append(v)
    Q_neg_max = np.array(values)
    # ic(values)

    # inference immediately updates to maximum probability from start
    # from equal probability
    P = [.5] + [1-p_switch] * n_trials
    P_equal = np.array(P)
    P_equal_values = inference_agent_value_update(P_equal, p_active_reward)
    # from negative maximum
    P = [0] + [1-p_switch] * n_trials
    P_neg_max = np.array(P)
    P_neg_max_values = inference_agent_value_update(P_neg_max, p_active_reward)
    return Q_zero, Q_neg_max, P_equal_values, P_neg_max_values


def main():
    plot_path = Path('../../reports/figures/F31_Apr2024')

    # Asymmetric inference parameters
    p_switch = .1
    p_active_reward = .9

    # Reinforcement learning parameters
    memory = .8
    omission_priors, omit_inference_values, qlearning_run = consecutive_omissions(p_switch, p_active_reward, memory, n_trials=20)
    f, ax = plt.subplots()
    # ax.plot(omission_priors, label='priors')
    ax.plot(omit_inference_values, 'k', label='Inference')
    ax.plot(qlearning_run, 'k--', label='RL')
    # plt.legend(fancybox=False)
    plt.legend(frameon=False, fontsize=16)
    plt.title('Consecutive Omissions', fontsize=20)
    plt.ylim([-1,1])
    plt.yticks([-1, -.5, 0, .5, 1])
    plt.xticks([0, 5, 10, 15, 20], [0, 5, 10, 15, 20])
    # plt.ylabel('Relative Value', fontsize=16)
    plt.xlabel('Trials', fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    height = 8
    width = 6
    f.set_figheight(height)
    f.set_figwidth(width)
    f.savefig(plot_path / 'consecutive_omissions.png', format='png', dpi=300)
    f.savefig(plot_path / 'consecutive_omissions.svg', format='svg')

    Q_zero, Q_neg_max, P_equal, P_neg_max = consecutive_rewards(p_switch, p_active_reward, memory, n_trials=20)

    f, ax = plt.subplots()
    ax.plot(Q_neg_max, 'k--', label='RL')
    ax.plot(P_neg_max, 'k', label='Inference')
    # plt.legend(fancybox=False)
    plt.title('Consecutive Rewards', fontsize=20)
    plt.ylabel('Relative Value', fontsize=20)
    plt.xlabel('Trials', fontsize=20)
    plt.ylim([-1, 1])
    plt.yticks([-1, -.5, 0, .5, 1])
    plt.xticks([0, 5, 10, 15, 20], [0, 5, 10, 15, 20])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    f.set_figheight(height)
    f.set_figwidth(width)
    f.savefig(plot_path / 'consecutive_rewards.png', format='png', dpi=300)
    f.savefig(plot_path / 'consecutive_rewards.svg', format='svg')

    # data = {'symmetric': symmetric_runs, 'asymmetric': asymmetric_runs, 'sym_switch': sym_switch,
    #         'asym_switch': asym_switch, 'p_active_reward': p_active_reward, 'p_switch': p_switch}
    note = ('symmetric and asymmetric matrices represent the probability of staying at a chosen port/side after a '
            'reward omission. Matrices are N x R x T, where N is the number of p_switch values, R is the number of '
            'p_active_reward values, and T is the number of trials.')

    # plt.show()
    # save_runs(data, 'inference_consecutive_omissions', note=note)


if __name__ == '__main__':
    main()
