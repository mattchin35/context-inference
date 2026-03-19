from pathlib import Path
import json
import pickle as pkl
from typing import Dict, Optional, Tuple
import logging
from collections import defaultdict
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.behavior_modeling.agents import agents
from src.behavior_modeling.parameters import task_config
from src.behavior_modeling.task.context_task import BaseMDP

SEED = 12345  # 0
rng = np.random.default_rng(SEED)
SUPPORTED_SAMPLE_AGENTS = {
    'HMM': 'HMM',
    'HMM_reward_decay': 'HMMdecay',
    'HMM_reward_decay_relative_doubt': 'HMMdecayDoubt',
    'F-Qlearning': 'FQL',
    'Qlearning': 'QL',
    'Logistic': 'RFLR',
}

TRIAL_DF_COLUMNS = [
    'state',
    'state_int',
    'cur_trial',
    'cur_trial_in_block',
    'cur_block',
    'action',
    'correct',
    'reward',
    'p_active_rew',
    'p_inactive_rew',
    'p_switch',
    'model_stimulus',
    'agent_action_dist',
    'agent_hmm_value',
    'agent_doubt_value',
    'agent_relative_value',
    'agent_prior',
]


@dataclass
class RunOutputOptions:
    save_run: bool = False
    save_plot: bool = False
    show_plot: bool = False
    run_data_dir: Optional[Path] = None
    plot_path: Optional[Path] = None
    value_columns: Optional[list[str]] = None
    theme: str = "light"

    @property
    def generate_plot(self) -> bool:
        return self.save_plot or self.show_plot


@dataclass
class RunExecutionOptions:
    seed: Optional[int] = None


@dataclass
class RunArtifacts:
    performance_df: pd.DataFrame
    run_path: Optional[Path] = None
    plot_path: Optional[Path] = None
    figure: Optional[object] = None
    params_path: Optional[Path] = None
    seed: Optional[int] = None


def get_agent_prior(agent: agents.BehaviorAgent) -> float:
    if isinstance(agent, agents.HMM):
        return agent.prior[1]
    return np.nan


def get_agent_value_component(agent: agents.BehaviorAgent, attr_name: str) -> float:
    value = getattr(agent, attr_name, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def performance_to_dataframe(performance: Dict[str, list]) -> pd.DataFrame:
    df = pd.DataFrame(performance)
    return df.reindex(columns=TRIAL_DF_COLUMNS)


def format_param_value(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):g}"
    return str(value)


def get_sample_agent_label(agent_name: str) -> str:
    if agent_name not in SUPPORTED_SAMPLE_AGENTS:
        raise ValueError(f"Unsupported sample-agent type: {agent_name}")
    return SUPPORTED_SAMPLE_AGENTS[agent_name]


def get_session_timestamp() -> str:
    return dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')


def create_run_rngs(seed: Optional[int] = None) -> tuple[Optional[np.random.Generator], Optional[np.random.Generator]]:
    if seed is None:
        return None, None

    seed_sequence = np.random.SeedSequence(seed)
    task_seed, agent_seed = seed_sequence.spawn(2)
    return np.random.default_rng(task_seed), np.random.default_rng(agent_seed)


def format_sample_agent_parameter_summary(
    agent_name: str,
    task_params: task_config.TaskParams,
    agent_params: task_config.AgentParams,
) -> str:
    fields = [
        f"pRew-{format_param_value(task_params.active_reward_probability)}",
        f"pInc-{format_param_value(task_params.inactive_reward_probability)}",
        f"pSwitch-{format_param_value(task_params.state_transition_prob)}",
        f"nTrials-{format_param_value(task_params.n_trials)}",
    ]

    if agent_name == 'HMM':
        fields.extend([
            f"mode-{agent_params.HMM_value_mode}",
            f"modelPsw-{format_param_value(agent_params.HMM_transition_prob)}",
            f"alpha-{format_param_value(agent_params.logistic_alpha)}",
            f"temp-{format_param_value(agent_params.action_temperature)}",
        ])
        if agent_params.HMM_value_mode == 'bayesian_log_odds':
            fields.append(f"tanh-{format_param_value(agent_params.HMM_log_odds_tanh_scale)}")
    elif agent_name == 'HMM_reward_decay':
        fields.extend([
            f"mode-{agent_params.HMM_value_mode}",
            f"modelPsw-{format_param_value(agent_params.HMM_transition_prob)}",
            f"alpha-{format_param_value(agent_params.logistic_alpha)}",
            f"temp-{format_param_value(agent_params.action_temperature)}",
            f"lambda-{format_param_value(agent_params.HMM_reward_decay_lambda)}",
        ])
        if agent_params.HMM_value_mode == 'bayesian_log_odds':
            fields.append(f"tanh-{format_param_value(agent_params.HMM_log_odds_tanh_scale)}")
    elif agent_name == 'HMM_reward_decay_relative_doubt':
        fields.extend([
            f"mode-{format_param_value(agent_params.HMM_value_mode)}",
            f"modelPsw-{format_param_value(agent_params.HMM_transition_prob)}",
            f"alpha-{format_param_value(agent_params.logistic_alpha)}",
            f"temp-{format_param_value(agent_params.action_temperature)}",
            f"lambda-{format_param_value(agent_params.HMM_reward_decay_lambda)}",
            f"doubtLam-{format_param_value(agent_params.relative_doubt_lambda)}",
        ])
        if agent_params.HMM_value_mode == 'bayesian_log_odds':
            fields.append(f"tanh-{format_param_value(agent_params.HMM_log_odds_tanh_scale)}")
    elif agent_name == 'F-Qlearning':
        fields.extend([
            f"decay-{format_param_value(agent_params.FQL_decay)}",
            f"alpha-{format_param_value(agent_params.logistic_alpha)}",
            f"temp-{format_param_value(agent_params.action_temperature)}",
        ])
    elif agent_name == 'Qlearning':
        fields.extend([
            f"lr-{format_param_value(agent_params.QL_learning_rate)}",
            f"temp-{format_param_value(agent_params.action_temperature)}",
        ])
    elif agent_name == 'Logistic':
        fields.extend([
            f"alpha-{format_param_value(agent_params.logistic_alpha)}",
            f"beta-{format_param_value(agent_params.logistic_beta)}",
            f"tau-{format_param_value(agent_params.logistic_tau)}",
        ])
    else:
        raise ValueError(f"Unsupported sample-agent type: {agent_name}")

    return '_'.join(fields)


def build_sample_agent_session_name(
    agent_name: str,
    task_params: task_config.TaskParams,
    agent_params: task_config.AgentParams,
    session_timestamp: str,
) -> str:
    model_label = get_sample_agent_label(agent_name)
    param_summary = format_sample_agent_parameter_summary(agent_name, task_params, agent_params)
    return f"{model_label}_{session_timestamp}_{param_summary}"


def build_run_plot_title(
    agent_name: str,
    task_params: task_config.TaskParams,
    agent_params: task_config.AgentParams,
) -> str:
    summary = format_agent_parameter_summary(agent_name, task_params, agent_params)
    return f"{agent_name} run\n{summary}"


def format_agent_parameter_summary(
    agent_name: str,
    task_params: task_config.TaskParams,
    agent_params: task_config.AgentParams,
) -> str:
    try:
        return format_sample_agent_parameter_summary(agent_name, task_params, agent_params)
    except ValueError:
        return (
            f"agent-{agent_name}_"
            f"pRew-{format_param_value(task_params.active_reward_probability)}_"
            f"pInc-{format_param_value(task_params.inactive_reward_probability)}_"
            f"pSwitch-{format_param_value(task_params.state_transition_prob)}_"
            f"nTrials-{format_param_value(task_params.n_trials)}"
        )


def build_run_stem(
    agent_name: str,
    task_params: task_config.TaskParams,
    agent_params: task_config.AgentParams,
    *,
    session_note: str = '',
    include_timestamp: bool = False,
) -> str:
    label = SUPPORTED_SAMPLE_AGENTS.get(agent_name, agent_name)
    param_summary = format_agent_parameter_summary(agent_name, task_params, agent_params)
    parts = [label]
    if include_timestamp:
        parts.append(get_session_timestamp())
    parts.append(param_summary)
    if session_note:
        parts.append(session_note)
    return '_'.join(parts)


def resolve_run_output_paths(
    agent_name: str,
    task_params: task_config.TaskParams,
    agent_params: task_config.AgentParams,
    output_options: RunOutputOptions,
    *,
    sample_agent_data_dir: Path,
    experiment_data_dir: Path,
    session_note: str = '',
) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    run_path = None
    params_path = None
    plot_path = output_options.plot_path
    is_sample_agent = agent_name in SUPPORTED_SAMPLE_AGENTS
    run_root = output_options.run_data_dir or (
        sample_agent_data_dir if is_sample_agent else experiment_data_dir
    )

    if is_sample_agent:
        run_stem = build_run_stem(
            agent_name,
            task_params,
            agent_params,
            session_note=session_note,
            include_timestamp=True,
        )
        output_dir = run_root / run_stem
        if output_options.save_run:
            run_path = output_dir / f"{run_stem}.csv"
            params_path = output_dir / f"{run_stem}_params.json"
        if output_options.save_plot and plot_path is None:
            plot_path = output_dir / f"{run_stem}_plot.png"
    else:
        run_stem = build_run_stem(
            agent_name,
            task_params,
            agent_params,
            session_note=session_note,
        )
        output_dir = run_root / str(dt.date.today().isoformat())
        if output_options.save_run:
            run_path = output_dir / f"{run_stem}.pkl"
        if output_options.save_plot and plot_path is None:
            plot_path = output_dir / f"{run_stem}_plot.png"

    return run_path, params_path, plot_path


def save_sample_agent_run(
    agent_name: str,
    performance_df: pd.DataFrame,
    task_params: task_config.TaskParams,
    agent_params: task_config.AgentParams,
    csv_path: Path,
    params_path: Path,
    run_seed: Optional[int] = None,
) -> tuple[Path, Path]:
    csv_path.parent.mkdir(parents=True, exist_ok=False)
    path_parts = csv_path.stem.split('_', 2)
    session_timestamp = path_parts[1] if len(path_parts) > 2 else get_session_timestamp()
    performance_df.to_csv(csv_path, index=False)

    params_payload = {
        'agent_name': agent_name,
        'model_type_label': SUPPORTED_SAMPLE_AGENTS.get(agent_name, agent_name),
        'session_timestamp': session_timestamp,
        'run_seed': run_seed,
        'task_params': vars(task_params),
        'agent_params': vars(agent_params),
        'dataframe_columns': list(performance_df.columns),
    }
    with params_path.open('w') as f:
        json.dump(params_payload, f, indent=2)
    return csv_path, params_path


def plot_run_dataframe(
    performance_df: pd.DataFrame,
    agent_name: str,
    task_params: task_config.TaskParams,
    agent_params: task_config.AgentParams,
    *,
    plot_save_path: Optional[Path] = None,
    show_plot: bool = False,
    value_columns: Optional[list[str]] = None,
    theme: str = "light",
):
    from src.behavior_modeling.visualize_behavior.agent_run_plot import (
        plot_run_dataframe as plot_agent_run_dataframe,
    )

    plot_title = build_run_plot_title(agent_name, task_params, agent_params)
    return plot_agent_run_dataframe(
        performance_df,
        title=plot_title,
        value_columns=value_columns,
        save_path=plot_save_path,
        show=show_plot,
        theme=theme,
    )


def run_task_cycle(task: BaseMDP, agent: agents.BehaviorAgent, performance: Dict[str, list]) -> \
        Tuple[BaseMDP, agents.BehaviorAgent, Dict[str, list]]:

    performance['state'].append(task.cur_state)
    performance['state_int'].append(task.state_dict[task.cur_state])
    performance['cur_trial'].append(task.cur_trial)
    performance['cur_block'].append(task.cur_block)
    performance['cur_trial_in_block'].append(task.cur_trial_in_block)
    performance['agent_relative_value'].append(agent.value)
    performance['p_active_rew'].append(task.params.active_reward_probability)
    performance['p_inactive_rew'].append(task.params.inactive_reward_probability)
    performance['p_switch'].append(task.params.state_transition_prob)
    performance['agent_prior'].append(get_agent_prior(agent))
    performance['agent_hmm_value'].append(get_agent_value_component(agent, 'hmm_value'))
    performance['agent_doubt_value'].append(get_agent_value_component(agent, 'doubt_value'))

    stimulus = task.get_stimulus()
    action, action_dist = agent.choose_action(stimulus)
    reward, correct = task.step(action)

    performance['model_stimulus'].append(stimulus)
    performance['agent_action_dist'].append(action_dist[1])
    agent.update_params(action, reward)

    # this will get refactored out to a controller class/fxn
    performance['action'].append(action)
    performance['correct'].append(correct)
    performance['reward'].append(reward)
    return task, agent, performance


def run_experiment(task: BaseMDP, agent: agents.BehaviorAgent, task_params: task_config.TaskParams) -> \
        Tuple[BaseMDP, agents.BehaviorAgent, Dict[str, list]]:

    performance = defaultdict(list)
    while task.cur_trial < task_params.n_trials:
        logging.info("block {}, trial {}".format(task.cur_block, task.cur_trial))
        task, agent, performance = run_task_cycle(task, agent, performance)

    return task, agent, performance


def run_agent_session(
    task: BaseMDP,
    agent: agents.BehaviorAgent,
    task_params: task_config.TaskParams,
) -> tuple[BaseMDP, agents.BehaviorAgent, pd.DataFrame]:
    task, agent, performance = run_experiment(task, agent, task_params)
    return task, agent, performance_to_dataframe(performance)


def select_agent(
    agent_name: str,
    agent_params: task_config.AgentParams,
    task_params: task_config.TaskParams,
    rng: Optional[np.random.Generator] = None,
) -> agents.BehaviorAgent:
    if agent_name == 'HMM':
        agent = agents.HMM(agent_params, task_params, rng=rng)
    elif agent_name == 'HMM_reward_decay':
        agent = agents.HMMRewardDecay(agent_params, task_params, rng=rng)
    elif agent_name == 'HMM_reward_decay_relative_doubt':
        agent = agents.HMMRewardDecayRelativeDoubt(agent_params, task_params, rng=rng)
    elif agent_name == 'HMM_RFLR':
        agent = agents.HMM_RFLR(agent_params, task_params, rng=rng)
    elif agent_name == 'HMM_recursive':
        agent = agents.HMM_recursive(agent_params, task_params, rng=rng)
    elif agent_name == 'Qlearning':
        agent = agents.Qlearning(agent_params, rng=rng)
    elif agent_name == 'F-Qlearning':
        agent = agents.ForgettingQlearning(agent_params, rng=rng)
    elif agent_name == 'Logistic':
        agent = agents.Logistic(agent_params, task_params, rng=rng)
    else:
        raise ValueError("Unknown agent name: {}".format(agent_name))
    return agent


def set_params() -> tuple[task_config.TaskParams, task_config.AgentParams]:
    ## TASK PARAMETERS ##
    task_params = task_config.TaskParams()

    task_params.n_states = 2
    task_params.n_actions = 2

    # Task structure parameters, for markov and success_trigger. All between 0 and 1
    task_params.block_transition_style = 'success_trigger'  # 'success_trigger', 'markov', 'n_correct', 'fixed'
    task_params.p_cue = 0  # probability of cue being shown at a block transition
    task_params.t_cue = 2  # number of timesteps with cue present
    task_params.state_transition_prob = .1  # true task dynamics, for markov and success_trigger
    task_params.active_reward_probability = .9
    task_params.inactive_reward_probability = 0
    task_params.n_trials = 100

    # params for fixed block lengths
    task_params.default_fixed_block_length = 15
    task_params.fixed_block_length_variation = 0
    task_params.success_trials_to_block_transition = np.inf  # a maximum number of correct trials before a block transition is forced

    # reward size parameters
    task_params.mean_correct_reward = 1
    task_params.mean_incorrect_reward = 0
    task_params.reward_std_dev = 0
    task_params.ITI_lick_reward = 0  # for licking outside of go cue; don't think I'll use this
    task_params.wait_reward = 0  # for not licking during go cue. Pick 0 (no waiting penalty) or -1 (waiting penalty)


    ## AGENT PARAMETERS ##
    agent_params = task_config.AgentParams()

    # for the HMM agent, you can play with using model parameters that are different from the true task parameters
    agent_params.HMM_transition_prob = task_params.state_transition_prob  # HMM agent's belief of state change probability
    agent_params.HMM_value_mode = 'expected_reward'
    agent_params.HMM_log_odds_tanh_scale = 1.0
    # agent_params.HMM_transition_prob = .1  # HMM agent's belief of task dynamics
    agent_params.HMM_active_reward_probability = .9  # HMM agent's belief of reward probability
    agent_params.HMM_inactive_reward_probability = 0  # HMM agent's belief of reward probability

    # Q-learning parameters
    agent_params.FQL_decay = .7  # for forgetting Q-learning agent
    agent_params.QL_learning_rate = .3  # for standard Q-learning agent
    agent_params.action_temperature = .3  # .3 seems to work for FQL  ## temp=1 is default; temp->0 makes greedier; temp->inf increases randomness
    agent_params.action_stickiness = 0  # tendency to repeat last action
    agent_params.greedy_action_selection = False
    agent_params.greedy_epsilon = .1

    # Logistic (and HMM-sticky) parameters
    agent_params.logistic_alpha = 1  # default 1 - action stickiness
    agent_params.logistic_beta = 2  # default 2 - action/reward history update size
    agent_params.logistic_tau = 1.5  # default 1.5 - memory for prior timestep settings, higher tau increases memory. 1.5 -> .5 memory

    # params.action_stickiness = 0  # use alpha instead. default for RFLR is 1; default for others is 0
    return task_params, agent_params


def save_experiment(save_name: str, data_dir: Path, task: BaseMDP, agent: agents.BehaviorAgent, performance: dict,
                    agent_params: task_config.AgentParams, task_params: task_config.TaskParams) -> Path:
    exp = dict(task=vars(task), agent=vars(agent), agent_params=vars(agent_params), task_params=vars(task_params),
               performance=performance)
    date = str(dt.date.today().isoformat())
    save_dir = data_dir / date
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    p = save_dir / (save_name + '.pkl')
    with p.open('wb') as f:
        pkl.dump(exp, f)

    # print("[***] Experiment saved as: {}".format(p.name))
    return p


def save_experiment_to_path(
    save_path: Path,
    task: BaseMDP,
    agent: agents.BehaviorAgent,
    performance: pd.DataFrame,
    agent_params: task_config.AgentParams,
    task_params: task_config.TaskParams,
    run_seed: Optional[int] = None,
) -> Path:
    exp = dict(
        task=vars(task),
        agent=vars(agent),
        agent_params=vars(agent_params),
        task_params=vars(task_params),
        performance=performance,
        run_seed=run_seed,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('wb') as f:
        pkl.dump(exp, f)
    return save_path


def save_run_outputs(
    task: BaseMDP,
    agent: agents.BehaviorAgent,
    performance_df: pd.DataFrame,
    agent_name: str,
    task_params: task_config.TaskParams,
    agent_params: task_config.AgentParams,
    run_path: Path,
    params_path: Optional[Path] = None,
    run_seed: Optional[int] = None,
) -> tuple[Path, Optional[Path]]:
    if agent_name in SUPPORTED_SAMPLE_AGENTS:
        if params_path is None:
            raise ValueError("params_path is required when saving sample-agent outputs.")
        return save_sample_agent_run(
            agent_name,
            performance_df,
            task_params,
            agent_params,
            run_path,
            params_path,
            run_seed=run_seed,
        )

    saved_path = save_experiment_to_path(
        run_path,
        task,
        agent,
        performance_df,
        agent_params,
        task_params,
        run_seed=run_seed,
    )
    return saved_path, None


def plot_run_outputs(
    performance_df: pd.DataFrame,
    agent_name: str,
    task_params: task_config.TaskParams,
    agent_params: task_config.AgentParams,
    output_options: RunOutputOptions,
    *,
    plot_path: Optional[Path],
):
    return plot_run_dataframe(
        performance_df,
        agent_name,
        task_params,
        agent_params,
        plot_save_path=plot_path if output_options.save_plot else None,
        show_plot=output_options.show_plot,
        value_columns=output_options.value_columns,
        theme=output_options.theme,
    )


def handle_run_outputs(
    task: BaseMDP,
    agent: agents.BehaviorAgent,
    performance_df: pd.DataFrame,
    agent_name: str,
    task_params: task_config.TaskParams,
    agent_params: task_config.AgentParams,
    output_options: RunOutputOptions,
    *,
    sample_agent_data_dir: Path,
    experiment_data_dir: Path,
    session_note: str = '',
    run_seed: Optional[int] = None,
) -> RunArtifacts:
    run_path, params_path, plot_path = resolve_run_output_paths(
        agent_name,
        task_params,
        agent_params,
        output_options,
        sample_agent_data_dir=sample_agent_data_dir,
        experiment_data_dir=experiment_data_dir,
        session_note=session_note,
    )
    artifacts = RunArtifacts(
        performance_df=performance_df,
        run_path=run_path,
        plot_path=plot_path,
        params_path=params_path,
        seed=run_seed,
    )

    if output_options.save_run and run_path is not None:
        saved_run_path, saved_params_path = save_run_outputs(
            task,
            agent,
            performance_df,
            agent_name,
            task_params,
            agent_params,
            run_path,
            params_path=params_path,
            run_seed=run_seed,
        )
        artifacts.run_path = saved_run_path
        artifacts.params_path = saved_params_path
        print("[***] Agent run saved to: {}".format(saved_run_path.resolve()))
        if saved_params_path is not None:
            print("[***] Agent params saved to: {}".format(saved_params_path.resolve()))

    if output_options.generate_plot:
        figure, _ = plot_run_outputs(
            performance_df,
            agent_name,
            task_params,
            agent_params,
            output_options,
            plot_path=plot_path,
        )
        artifacts.figure = figure
        if output_options.save_plot and plot_path is not None:
            print("[***] Plot saved to: {}".format(plot_path.resolve()))
        elif output_options.show_plot:
            print("[***] Plot displayed (not saved to disk).")

    return artifacts


def load_experiment(load_name: str) -> Tuple[BaseMDP, agents.BehaviorAgent, pd.DataFrame, task_config.TaskParams, task_config.AgentParams]:
    with open(load_name, 'rb') as f:
        exp = pkl.load(f)

    task_params = task_config.TaskParams()
    for k, v in exp['task_params'].items():
        setattr(task_params, k, v)

    agent_params = task_config.AgentParams()
    for k, v in exp['agent_params'].items():
        setattr(agent_params, k, v)

    task = BaseMDP(task_params)
    agent = agents.BehaviorAgent

    for k, v in exp['task'].items():
        setattr(task, k, v)

    for k, v in exp['agent'].items():
        setattr(agent, k, v)

    print("[***] Model restored from path: {}".format(load_name))
    return task, agent, exp['performance'], task_params, agent_params


def main():
    data_dir = Path('../../data/processed/model_experiments')
    sample_agent_data_dir = Path('../../data/processed/sample_agents')
    # model_dir = Path('../../saved_models/standard_models')
    # figure_dir = Path('../../figures')

    log_path = Path('collection_task.log')
    log_path.unlink(missing_ok=True)
    logging.basicConfig(format='%(message)s', filename='task.log', level=logging.INFO)

    task_params, agent_params = set_params()
    session_note = ''
    execution_options = RunExecutionOptions(seed=0)
    output_options = RunOutputOptions(
        save_run=True,
        save_plot=True,
        show_plot=False,
    )

    # agent_name = 'F-Qlearning'
    # agent_param_str = 'alpha={}_temp={}_decay={}'.format(params.logistic_alpha, params.action_temperature, params.FQL_decay)

    agent_name = 'HMM'

    # agent_name = 'HMM_recursive'
    # agent_param_str = 'alpha={}_temp={}_model-Psw={}'.format(params.logistic_alpha, params.action_temperature, params.HMM_transition_prob)

    # agent_name = 'HMM_RFLR'
    # agent_param_str = 'alpha={}_beta={}_tau={}'.format(agent_params.logistic_alpha, agent_params.logistic_beta,
    #                                                    agent_params.logistic_tau)

    # agent_name = 'Logistic'
    # agent_param_str = 'alpha={}_beta={}_tau={}'.format(agent_params.logistic_alpha, agent_params.logistic_beta,
    #                                                    agent_params.logistic_tau)

    # pick a file path based on agent type (HMM, Qlearning, etc.)
    # pick a filename based on task-specific params (pReward, pSwitch, rewards-before-state-switch, etc.)

    ### RUN A NEW EXPERIMENT ###
    if execution_options.seed is not None:
        print("[***] Using run seed: {}".format(execution_options.seed))
    task_rng, agent_rng = create_run_rngs(execution_options.seed)
    task = BaseMDP(task_params, rng=task_rng)
    agent = select_agent(agent_name, agent_params, task_params, rng=agent_rng)
    task, agent, performance_df = run_agent_session(task, agent, task_params)
    handle_run_outputs(
        task,
        agent,
        performance_df,
        agent_name,
        task_params,
        agent_params,
        output_options,
        sample_agent_data_dir=sample_agent_data_dir,
        experiment_data_dir=data_dir,
        session_note=session_note,
        run_seed=execution_options.seed,
    )

    ### REPEAT THE TRIALS FROM A MODEL SESSION ###
    # p_reward = .9
    # p_switch = .1
    # alpha = .5
    # beta = 2
    # tau = 1.5
    # temp = .2
    # decay = .7
    # HMM_transition_prob = .1

    # load_agent_name = 'F-Qlearning'
    # load_param_str = 'alpha={}_temp={}_decay={}'.format(alpha, temp, decay)
    # load_agent_name = 'HMM'
    # load_param_str = 'alpha={}_temp={}_model-Psw={}'.format(alpha, temp, HMM_transition_prob)
    # load_agent_name = 'Logistic'
    # load_param_str = 'alpha={}_beta={}_tau={}'.format(alpha, beta, tau)
    # extras_str = 'comparison-model'
    #
    # load_session_name = '{}_pReward_{}_pSwitch_{:.2f}'.format(load_agent_name, p_reward, p_switch)
    # load_fname = load_session_name + '_' + load_param_str
    # if extras_str:
    #     load_fname += '_' + extras_str
    # p = Path('../saved_models') / (load_fname + '.pkl')
    # _, _, task_df, _ = load_experiment(p)


if __name__ == '__main__':
    main()
