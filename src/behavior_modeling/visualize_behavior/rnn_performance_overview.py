from src.behavior_modeling import rnn_controller as rc
from src.behavior_modeling.fileIO import rnn_io
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data_dir = Path('../../../data/processed/rnn_experiments')
model_dir = Path('../../../saved_models/RNN')
plot_path = Path('../../../reports/figures/model_behavior')

states = ['right', 'left']
state_dict = {s: i for i, s in enumerate(states)}  # i.e. [0 right, 1 left]


def performance_overview():
    agent_name = 'RNN_reinforce'
    date = '2024-04-11'
    p_reward = .9
    p_switch = .1
    fixed_blocks = 10
    note = 'overtrain'

    load_name = '{}_pReward_{}_pSwitch_{}'.format(agent_name, p_reward, p_switch)
    if note:
        load_name += '_{}'.format(note)
    # load_name = '{}_pReward_{}_fixedBlocks_{}'.format(agent_name, p_reward, fixed_blocks)
    session_name = data_dir / date / (load_name + '_eval_dict.pkl')
    # session_name = data_dir / date / (load_name + '_dict.pkl')
    experiment = rnn_io.load_experiment(session_name)
    performance = experiment['performance']
    # state_dict = experiment['state_dict']

    states = performance['states']
    states_int = np.array([state_dict[s] for s in states])[:, None]
    # action = performance['action'].to_numpy()[:, None]
    # reward = performance['reward'].to_numpy()[:, None]
    action = np.array(performance['action'])[:, None]
    reward = np.array(performance['reward'])[:, None]
    correct = np.array(performance['correct'])[:, None]
    # stimulus = np.stack(performance['stimulus'].to_numpy(), axis=0)
    inputs = np.stack(experiment['rnn_dict']['inputs'], axis=0)
    go_cue = inputs[:, 1] > 0

    Rstimulus = inputs[:, 2] > 0
    Lstimulus = inputs[:, 3] > 0
    allstimulus = Rstimulus | Lstimulus
    ix = np.arange(inputs.shape[0])
    X = np.concatenate([ix[Rstimulus], ix[Lstimulus]])
    Y = np.concatenate([np.zeros(np.sum(Rstimulus)), np.ones(np.sum(Lstimulus))])

    f, ax = plt.subplots(1,4)
    plt.sca(ax[0])
    sns.heatmap(states_int[go_cue][:200])
    plt.title('States')
    plt.sca(ax[1])
    sns.heatmap(action[go_cue][:200])
    plt.title('Actions')
    plt.sca(ax[2])
    sns.heatmap(reward[go_cue][:200])
    plt.title('Rewards')
    plt.sca(ax[3])
    sns.heatmap(inputs[go_cue][:200,2:])
    # plt.imshow(inputs, aspect='auto', interpolation='none')
    plt.title('R cue/L cue')

    plt.tight_layout()
    save_path = plot_path / '{}_performance-overview.png'.format(load_name)
    f.savefig(save_path, format='png', dpi=300)
    print('Saved as {}'.format(save_path.resolve()))

    f2, ax2 = plt.subplots(1,1)
    # plt.plot(correct[go_cue][:100].flatten(), label='correct')
    # plt.plot(reward[go_cue][:100].flatten(), label='reward')
    # plt.plot(action[go_cue][:100].flatten(), label='action')

    plt.plot(action[go_cue][:100].flatten(), 'o', label='action')
    plt.plot(states_int[go_cue][:100].flatten(), label='states')
    plt.plot(allstimulus[go_cue][:100], label='allstimulus')

    # plt.plot(action[:200].flatten(), 'o', label='action')
    # plt.plot(states_int[:200].flatten(), label='states')
    # plt.plot(allstimulus[:200], label='allstimulus')

    # stim_ix = X < 100
    # plt.plot(X[stim_ix], Y[stim_ix], 'x', label='stimulus', markersize=20)
    plt.legend()

    # f3, ax3 = plt.subplots(1, 2)
    # plt.sca(ax3[0])
    # sns.heatmap(inputs[:100,1][None,:])
    # plt.sca(ax3[1])
    # plt.plot(go_cue[:100].flatten())
    # plt.suptitle('Go cue')
    plt.tight_layout()
    save_path = plot_path / '{}_performance-zoom.png'.format(load_name)
    f2.savefig(save_path, format='png', dpi=300)

    # plt.show()


def agent_overview():
    load_name = '{}_pReward_{}_pSwitch_{}'.format(agent_name, .9, .1)
    date = '2024-04-10'
    load_checkpoint = model_dir / date / (load_name + '_agent.pkl')
    # load_checkpoint = ''
    if load_checkpoint:
        print('Loading from checkpoint {}'.format(load_checkpoint))
        agent.load_state_dict(torch.load(load_checkpoint))


def main():
    performance_overview()
    # agent_overview()


if __name__ == '__main__':
    main()

