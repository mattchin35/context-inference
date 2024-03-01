import numpy as np
from typing import Tuple, List, Any, Iterable, Dict
import logging
import datetime as dt
import pickle as pkl
from pathlib import Path
import matplotlib.pyplot as plt


EPS = np.finfo(float).eps
figure_dir = Path('../../reports/figures')


def load_inference_demo(date) -> Tuple[Dict, str]:
    fname = Path('../../data/processed/{}/consecutive_omissions/consecutive_omissions.pkl'.format(date))
    with open(fname, 'rb') as f:
        data = pkl.load(f)

    dataset_name = '{}_inference-consecutive-omissions'.format(date)
    return data, dataset_name


def plot_heatmap(p_stay: np.ndarray, p_active_reward: np.ndarray, p_switch: float, title: str, tag: str) -> None:
    f, ax = plt.subplots(1, 1)
    im_sym = ax.imshow(p_stay, vmin=0, vmax=1)
    plt.title('{} heatmap of P(stay); P(switch) = {:.1f}\ntag={}'.format(title, p_switch, tag))

    plt.sca(ax)
    plt.ylabel('P(active reward)')
    plt.yticks(ticks=range(len(p_active_reward)), labels=['{:.1f}'.format(p) for p in p_active_reward])
    plt.xticks(np.arange(10), np.arange(10) + 1)
    cbar_sym = ax.figure.colorbar(im_sym, ax=ax)

    plt.tight_layout()
    savefig(f, '{}_heatmap_pswitch_{:.1f}'.format(title, p_switch))
    plt.close('all')


def plot_relative_value_heatmap(p_stay: np.ndarray, decay: np.ndarray, title: str, tag: str) -> None:
    f, ax = plt.subplots(1, 1)
    im_sym = ax.imshow(p_stay, vmin=0, vmax=1)
    plt.title('{} heatmap of P(stay); \ntag={}'.format(title, tag))

    plt.sca(ax)
    plt.ylabel('Decay')
    plt.yticks(ticks=range(len(decay)), labels=['{:.1f}'.format(1-d) for d in decay])
    plt.xlabel('Trials')
    plt.xticks(np.arange(10), np.arange(10) + 1)
    cbar_sym = ax.figure.colorbar(im_sym, ax=ax)

    plt.tight_layout()
    savefig(f, '{}_heatmap'.format(title))
    plt.close('all')


def plot_all_heatmaps(data: Dict, tag: str) -> None:
    """
    Plot the heatmap of the data.
    """

    symmetric = data['symmetric']
    asymmetric = data['asymmetric']
    recursion = data['recursion']
    p_active_reward = data['p_active_reward']
    p_switch = data['p_switch']

    qlearning = data['qlearning']
    decay = data['decay']

    for i, p in enumerate(p_switch):
        plot_heatmap(symmetric[i], p_active_reward, p, 'Symmetric', tag)
        plot_heatmap(asymmetric[i], p_active_reward, p, 'Asymmetric', tag)
        plot_heatmap(recursion[i], p_active_reward, p, 'Recursion', tag)

    plot_relative_value_heatmap(p_stay=qlearning, decay=decay, title='Q-learning relative value', tag=tag)  # this isn't quite accurate, pstay is relative value but not probability


def savefig(figure: plt.Figure, name: str, dpi=300) -> None:
    date = str(dt.date.today().isoformat())
    save_dir = Path('{}/{}/demo_runs'.format(figure_dir, date))
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    p = save_dir / (name + '.png')
    figure.savefig(p, dpi=dpi)


def plot_all_trials_to_switch(data: Dict, tag: str) -> None:
    sym_switch = data['sym_switch']
    asym_switch = data['asym_switch']
    rec_switch = data['rec_switch']
    p_active_reward = data['p_active_reward']
    p_switch = data['p_switch']

    for i, p in enumerate(p_switch):
        plot_trials_to_switch(sym_switch[i], p_active_reward, p, 'Symmetric', tag)
        plot_trials_to_switch(asym_switch[i], p_active_reward, p, 'Asymmetric', tag)
        plot_trials_to_switch(rec_switch[i], p_active_reward, p, 'Recursion', tag)
    plot_trials_to_switch(data['RL_switch'], 1-data['decay'], label='Q-learning', tag=tag, mode='qlearning')


def plot_trials_to_switch(switch: np.ndarray, p_active_reward: np.ndarray, p_switch: float=None,
                          label: str=None, tag: str=None, mode='inference') -> None:
    f, ax = plt.subplots()
    plt.plot(p_active_reward, switch)
    plt.ylabel('Trials to switch')
    if mode == 'inference':
        plt.xlabel('P(active reward)')
        plt.title('{} trials to switch as a function of P(active reward); P(switch) = {:.1f}\ntag={}'.format(label, p_switch, tag))
        savefig(f, '{}_trials_to_switch_pswitch_{:.1f}'.format(label, p_switch))
    elif mode == 'qlearning':
        plt.xlabel('Decay')
        plt.title('{} trials to switch as a function of decay\ntag={}'.format(label, tag))
        savefig(f, '{}_trials_to_switch'.format(label))

    plt.close('all')


def main():
    data, dataset_name = load_inference_demo('2024-01-08')
    plot_all_heatmaps(data, dataset_name)
    plot_all_trials_to_switch(data, dataset_name)
    # plt.show()


if __name__ == '__main__':
    main()
