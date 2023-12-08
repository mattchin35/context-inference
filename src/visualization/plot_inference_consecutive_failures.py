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
    """
    Load data from the inference demo.
    """
    fname = Path('../../data/processed/{}/demo_runs/inference_consecutive_failures.pkl'.format(date))
    with open(fname, 'rb') as f:
        data = pkl.load(f)

    dataset_name = '{}_inference-consecutive-failures'.format(date)
    return data, dataset_name


def plot_heatmap(data: Dict, tag: str) -> None:
    """
    Plot the heatmap of the data.
    """

    symmetric = data['symmetric']
    asymmetric = data['asymmetric']
    p_active_reward = data['p_active_reward']
    p_switch = data['p_switch']

    f, ax = plt.subplots(2, 1)
    im_sym = ax[0].imshow(symmetric, vmin=0, vmax=1)
    im_asym = ax[1].imshow(asymmetric, vmin=0, vmax=1)
    plt.suptitle('Heatmap of P(stay); P(switch) = {}\ntag={}'.format(p_switch, tag))

    plt.sca(ax[0])
    plt.title('Symmetric transition matrix')
    plt.ylabel('P(active reward)')
    plt.yticks(range(len(p_active_reward)), p_active_reward)
    plt.xticks(np.arange(10), np.arange(10)+1)
    cbar_sym = ax[0].figure.colorbar(im_sym, ax=ax[0])

    plt.sca(ax[1])
    plt.title('Asymmetric transition matrix')
    plt.xlabel('Trial')
    plt.ylabel('P(active reward)')
    plt.yticks(range(len(p_active_reward)), p_active_reward)
    plt.xticks(range(10), range(10))
    cbar_asym = ax[1].figure.colorbar(im_sym, ax=ax[1])

    plt.tight_layout()
    savefig(f, 'heatmap_pstay')


def savefig(figure: plt.Figure, name: str, dpi=300) -> None:
    date = str(dt.date.today().isoformat())
    save_dir = Path('{}/{}/demo_runs'.format(figure_dir, date))
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    p = save_dir / (name + '.png')
    figure.savefig(p, dpi=dpi)


def plot_trials_to_switch(data: Dict, tag: str) -> None:
    sym_switch = data['sym_switch']
    asym_switch = data['asym_switch']
    p_active_reward = data['p_active_reward']

    f, ax = plt.subplots()
    plt.plot(p_active_reward, sym_switch, label='symmetric')
    plt.plot(p_active_reward, asym_switch, label='asymmetric')
    plt.legend()
    plt.ylabel('Trials to switch')
    plt.xlabel('P(active reward)')
    plt.title('Trials to switch as a function of P(active reward)\ntag={}'.format(tag))

    savefig(f, 'trials_to_switch')


def main():
    data, dataset_name = load_inference_demo('2023-12-06')
    plot_heatmap(data, dataset_name)
    plot_trials_to_switch(data, dataset_name)
    # plt.show()


if __name__ == '__main__':
    main()
