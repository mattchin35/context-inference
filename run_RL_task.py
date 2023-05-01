import numpy as np
import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt
import lever_behavior
import bandits


if __name__ == '__main__':
    experiment = lever_behavior.TwoLeverAction(block_structure=['A', 'B', 'C1', 'C2', 'B'])
    experiment.default_blocklength = 60

    experiment.run_experiment(alpha=.5)
    print('actions', experiment.actions)  # A
    print('rewards', experiment.rewards)  # R
    print('baseline', experiment.baseline[-1])  # Q
    print('history', experiment.history[-1])  # H
    print('policy', experiment.policy[-1])  # Hdist

    print(len(experiment.actions))
    f = plt.figure()
    # f, ax = plt.subplots(2,1)
    x = np.arange(len(experiment.actions))
    y = np.array(experiment.actions)
    # plt.plot(x, experiment.actions, label='action')
    plt.scatter(x, y, label='action', s=4)

    n=5
    y = experiment.moving_average(experiment.actions, n=5)
    x_moving = np.arange(y.size)+n-1
    # plt.plot(x_moving, y, label='moving average action, n={}'.format(n), c='m')
    plt.yticks([0,1],['Left', 'Right'])
    plt.ylim([-.1,1.1])
    plt.ylabel('Actions')
    plt.xlabel('Trial')
    plt.title('Reward-only model behavior')
    ax = plt.gca()
    # ax.legend()
    ax2 = ax.twinx()
    plt.sca(ax2)
    plt.plot(x, experiment.states, label='state', c='g')
    plt.ylabel('States')
    # plt.legend()

    format = 'png'
    f.tight_layout()
    plt.savefig('../figures/nonstationary_bandit_actions.' + format, format=format)

    # ax[1].plot
    f1 = plt.figure()
    Q = np.array(experiment.baseline)
    plt.plot(x, Q[:,0], label='value, left lever press')
    plt.plot(x, Q[:,1], label='value, right lever press')
    # plt.plot(x, Q[:,0], label='value, withhold')
    # plt.plot(x, experiment.rewards, label='reward')
    y = experiment.moving_average(experiment.rewards, n=5)
    plt.plot(x_moving, y, label='moving avg reward, n={}'.format(n))
    plt.legend()

    # f2 = plt.figure()
    # Hdist = np.array(experiment.policy)
    # H = np.array(experiment.history)
    # plt.plot(x, Hdist[:,1], label='prob lever press')
    # plt.plot(x, H[:,1], label='pre-softmax lever press')
    # # plt.plot(x, H[:,0], label='pre-softmax withhold')
    # # plt.ylim([0,1])
    # # plt.plot(x, Q[:, 0], label='value, withhold')
    # plt.legend()
    f.tight_layout()
    plt.savefig('../figures/nonstationary_bandit_rewards.' + format, format=format)
    # plt.show()
