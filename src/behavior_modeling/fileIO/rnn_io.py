import pickle as pkl
from pathlib import Path
import torch
import pandas as pd
import datetime as dt
from typing import Tuple
from src.behavior_modeling.parameters import task_config, rnn_config
from src.behavior_modeling.task import rnn_task
from src.behavior_modeling import rnn_controller as rc

"""
TODO
HMM/RL
- save performance dict
- load performance dict
- save agent
- load agent
"""

# data_dir = Path('../../../data/processed/rnn_experiments')
# model_dir = Path('../../../saved_models/RNN')


def save_experiment(save_name: str, data_dir: Path, task: rnn_task.RnnMDP, performance: dict, rnn_dict: dict, agent_name: str,
                    task_params: task_config.TaskParams, rnn_params: rnn_config.AgentConfig) -> None:
    exp_dict = dict(task=vars(task), task_params=vars(task_params), performance=performance, rnn_dict=rnn_dict,
                    agent_params=vars(rnn_params), agent_name=agent_name, state_dict=task.state_dict)
    date = str(dt.date.today().isoformat())
    save_dir = data_dir / date
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    p_dict = save_dir / (save_name + '_dict.pkl')
    with open(p_dict, 'wb') as f:
        pkl.dump(exp_dict, f)

    print("[***] Experiment saved as: {}".format(p_dict.resolve()))


def save_agent(save_name: str, model_dir: Path, agent: torch.nn.Module) -> None:
    date = str(dt.date.today().isoformat())
    save_dir = model_dir / date
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    p_agent = save_dir / (save_name + '_agent.pkl')
    torch.save(agent.state_dict(), p_agent)
    print("[***] Agent saved as: {}".format(p_agent.resolve()))


def load_experiment(exp_path: str) -> Tuple[task_config.TaskParams, rnn_config.AgentConfig, pd.DataFrame]:
    # p_dict = data_dir / (load_name + '_dict.pkl')
    with open(exp_path, 'rb') as f:
        exp = pkl.load(f)

    # task_params = task_config.TaskParams()
    # if exp['task_params']:
    #     # for k, v in vars(exp['task_params']).items():
    #     for k, v in exp['task_params'].items():
    #         setattr(task_params, k, v)
    #
    # rnn_params = rnn_config.AgentConfig()
    # if exp['agent_params']:
    #     for k, v in exp['agent_params'].items():
    #         setattr(rnn_params, k, v)

    print("[***] Loaded experiment from path: {}".format(exp_path))
    return exp


def load_agent_state(agent: torch.nn.Module, agent_path: str) -> torch.nn.Module:
    agent.load_state_dict(torch.load(agent_path))
    print("[***] Agent restored from path: {}".format(agent_path))
    return agent
