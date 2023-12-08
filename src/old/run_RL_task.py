import numpy as np
import matplotlib.pyplot as plt
import context_behavior
import pandas as pd
import seaborn as sns
import os


if __name__ == '__main__':

    ### RUN EXPERIMENTS ###
    block_length = 60
    block_structure = ['A', 'B', 'C1', 'C2']
    # block_structure = block_structure + block_structure
    nBlocks = len(block_structure)
    model_type = 'HMM'
    save_name1 = '{}'.format(model_type)
    experiment1 = context_behavior.TwoLeverAction(block_structure=block_structure, default_blocklength=block_length)
    experiment1.p_switch = 1 / block_length
    experiment1.run_experiment(save_name=save_name1, model_name=model_type)

    model_type = 'stateless'
    save_name2 = '{}'.format(model_type)
    experiment2 = context_behavior.TwoLeverAction(block_structure=block_structure, default_blocklength=block_length)
    experiment2.run_experiment(save_name=save_name2, model_name=model_type)

    ## how to load a previous run
    # experiment1 = lever_behavior.TwoLeverAction()
    # fname = '{}_fast'.format(model_type)
    # experiment1.load(fname)
    #
    # experiment2 = lever_behavior.TwoLeverAction()
    # fname = '{}_slow'.format(model_type)
    # experiment2.load(fname)

    # print('actions', experiment.actions)  # A
    # print('rewards', experiment.rewards)  # R
    # print('baseline', experiment.baseline[-1])  # Q
    # print('history', experiment.history[-1])  # H
    # print('policy', experiment.policy[-1])  # Hdist

    f = plt.figure()
    x = np.arange(len(experiment1.actions))
    y = np.array(experiment1.actions)
    yA = []
    for a in y:
        if a == 0:
            yA.append('Left')
        else:
            yA.append('Right')

    y2 = np.array(experiment2.actions)
    yB = []
    for b in y2:
        if b == 0:
            yB.append('Left')
        else:
            yB.append('Right')

    dFast = {'action': yA, 'trial': x, 'learning_rate': 'fast learning'}
    dfFast = pd.DataFrame(data=dFast)
    dSlow = {'action': yB, 'trial': x, 'learning_rate': 'slow learning'}
    dfSlow = pd.DataFrame(data=dSlow)
    df = pd.concat([dfFast, dfSlow])

    # plt.scatter(x, y, label='fast learning', s=4)
    # sns.swarmplot(df, x='trial', y='action', hue='learning_rate',s=4)#, dodge=True)
    sns.swarmplot(df, x='trial', y='action', hue='learning_rate',s=4, dodge=True)

    ### PLOT moving average actions
    n = 5
    y = experiment1.moving_average(experiment1.actions, n=5)
    x_moving = np.arange(y.size) + n - 1
    # plt.plot(x_moving, y, label='fast moving average action', c='m')

    # plt.scatter(x, experiment2.actions, label='slow learning', s=4)
    y2 = experiment1.moving_average(experiment2.actions, n=5)
    # plt.plot(x_moving, y2, label='slow moving average action')

    ### SCATTERPLOT BEHAVIOR
    # plt.yticks([0,1],['Left', 'Right'])
    # plt.ylim([-.1,1.1])
    plt.ylabel('Actions', fontsize=12)
    plt.xticks(np.array([0,1,2,3,4]) * block_length)
    plt.xlim([0,block_length*nBlocks])
    plt.xlabel('Trial', fontsize=12)
    plt.title('{} model behavior'.format(model_type), fontsize=14)
    ax = plt.gca()
    ax.legend(fancybox=False)#, loc='lower right')

    # ax2 = ax.twinx()
    # plt.sca(ax2)
    # plt.plot(x, experiment.states, label='context', c='g')
    # plt.ylabel('Context', fontsize=12)
    # plt.legend()

    # plt.axvspan(0,block_length, color='cyan',alpha=.2)
    # plt.axvspan(block_length,block_length*2, color='darkgreen',alpha=.2)
    # plt.axvspan(block_length*2,block_length*3, color='plum',alpha=.2)
    # plt.axvspan(block_length*3,block_length*4, color='indigo',alpha=.2)

    # for i in [0,1]:
    # plt.axvspan(0*4*i, block_length*4*i, color='cyan', alpha=.2)
    # plt.axvspan(block_length, block_length * 2, color='darkgreen', alpha=.2)
    # plt.axvspan(block_length * 2, block_length * 3, color='plum', alpha=.2)
    # plt.axvspan(block_length * 3, block_length * 4, color='indigo', alpha=.2)
    colors = ['cyan', 'darkgreen', 'plum', 'indigo'] * 2
    for i, c in enumerate(colors):
        plt.axvspan(block_length*i, block_length * (i+1), color=c, alpha=.2)

    # if keys_list[i].startswith('ContextA'):
    #     plt.axvspan(i - 0.5, i + 0.5, color='cyan', alpha=0.1)
    # if keys_list[i].startswith('ContextB'):
    #     plt.axvspan(i - 0.5, i + 0.5, color='mediumseagreen', alpha=0.1)
    # if keys_list[i].startswith('ContextC'):
    #     plt.axvspan(i - 0.5, i + 0.5, color='lightgray', alpha=0.01)

    format = 'png'
    f.tight_layout()
    # plt.savefig('../figures/nonstationary_bandit_actions_combined.' + format, format=format, dpi=300)
    # plt.savefig('../figures/stateless_actions_2speeds.' + format, format=format, dpi=300)
    # plt.savefig('../figures/stateQ_actions_combined.' + format, format=format, dpi=300)
    plt.savefig('../figures/{}_actions.'.format(model_type) + format, format=format, dpi=300)

    # f, ax = plt.subplots()
    # policy = np.array(experiment.policy)
    # baseline = np.array(experiment.baseline)
    # plt.plot(policy[:,0])
    # plt.ylabel('left choice prob')
    # ax2 = ax.twinx()
    # plt.sca(ax2)
    # plt.plot(baseline[:,0],c='g')
    # plt.ylabel('left reward')
    # plt.show()


    # ax[1].plot
    # f1 = plt.figure()
    # Q = np.array(experiment.baseline)
    # plt.plot(x, Q[:,0], label='value, left lever press')
    # plt.plot(x, Q[:,1], label='value, right lever press')
    # # plt.plot(x, Q[:,0], label='value, withhold')
    # # plt.plot(x, experiment.rewards, label='reward')
    # y = experiment.moving_average(experiment.rewards, n=5)
    # plt.plot(x_moving, y, label='moving avg reward, n={}'.format(n))
    # plt.legend()

    # f2 = plt.figure()
    # Hdist = np.array(experiment.policy)
    # H = np.array(experiment.history)
    # plt.plot(x, Hdist[:,1], label='prob lever press')
    # plt.plot(x, H[:,1], label='pre-softmax lever press')
    # # plt.plot(x, H[:,0], label='pre-softmax withhold')
    # # plt.ylim([0,1])
    # # plt.plot(x, Q[:, 0], label='value, withhold')
    # plt.legend()
    # f.tight_layout()
    # plt.savefig('../figures/nonstationary_bandit_rewards.' + format, format=format)
    # plt.show()
