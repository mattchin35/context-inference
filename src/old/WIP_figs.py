import numpy as np
import matplotlib.pyplot as plt
import context_behavior
import pandas as pd
import seaborn as sns
import os


if __name__ == '__main__':
    block_length = 60
    block_structure = ['A', 'B', 'C1', 'C2']
    nBlocks = len(block_structure)
    model_type = 'HMM'
    save_name1 = '{}'.format(model_type)
    experiment1 = context_behavior.TwoLeverAction(block_structure=block_structure, default_blocklength=block_length)
    experiment1.p_switch = 1/block_length
    experiment1.run_experiment(save_name=save_name1, model_name=model_type)

    model_type = 'forgetting_Q'
    save_name2 = '{}'.format(model_type)
    experiment2 = context_behavior.TwoLeverAction(block_structure=block_structure, default_blocklength=block_length)
    experiment2.Qalpha = .8
    experiment2.run_experiment(save_name=save_name2, model_name=model_type)

    ### LOAD A PREVIOUS RUN
    # experiment1 = lever_behavior.TwoLeverAction()
    # fname = '{}_fast'.format(model_type)
    # experiment1.load(fname)
    #
    # experiment2 = lever_behavior.TwoLeverAction()
    # fname = '{}_slow'.format(model_type)
    # experiment2.load(fname)

    ### CREATE PANDAS DATAFRAMES FOR SEABORN
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

    x = np.arange(len(experiment1.actions))
    dHMM = {'action': yA, 'trial': x, 'model_type': 'HMM'}
    dfHMM = pd.DataFrame(data=dHMM)
    dQ = {'action': yB, 'trial': x, 'model_type': 'Q-learning'}
    dfQ = pd.DataFrame(data=dQ)
    df = pd.concat([dfHMM, dfQ])

    ### PLOT MODEL ACTIONS
    f0 = plt.figure()
    # plt.scatter(x, y, label='fast learning', s=4)
    # sns.swarmplot(df, x='trial', y='action', hue='model_type',s=4)  # no dodge plotting
    sns.swarmplot(df, x='trial', y='action', hue='model_type', s=4, dodge=True)

    # FIGURE ADJUSTMENTS
    # plt.yticks([0,1],['Left', 'Right'])
    # plt.ylim([-.1,1.1])
    plt.ylabel('Actions', fontsize=12)
    plt.xticks(np.array([0,1,2,3,4]) * block_length)
    plt.xlim([0,block_length*nBlocks])
    plt.xlabel('Trial', fontsize=12)
    plt.title('HMM vs Q-learning model behavior', fontsize=14)
    ax = plt.gca()
    ax.legend(fancybox=False)#, loc='lower right')

    ### COLOR BLOCKS
    colors = ['cyan', 'darkgreen', 'plum', 'indigo']
    for i, c in enumerate(colors):
        plt.axvspan(block_length*i, block_length * (i+1), color=c, alpha=.2)

    format = 'png'
    f0.tight_layout()
    plt.show()
    plt.savefig('../figures/HMM-vs-Q_actions.' + format, format=format, dpi=300)
