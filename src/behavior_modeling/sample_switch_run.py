"""Sample switched-run entry point for visualizing behavior-model transitions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.behavior_modeling import controller
from src.behavior_modeling.agent_switching import switching
from src.behavior_modeling.parameters import task_config
from src.behavior_modeling.task.context_task import BaseMDP


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "reports/figures/model_behavior/switching"
DEFAULT_N_TRIALS_PER_SEGMENT = 25
DEFAULT_AGENT_SEQUENCE = (
    "HMM",
    "F-Qlearning",
    "HMM",
    "HMM_reward_decay_relative_doubt",
)


def build_default_sample_switch_schedule(
    n_trials_per_segment: int = DEFAULT_N_TRIALS_PER_SEGMENT,
) -> list[dict[str, int | str]]:
    """Build the default four-segment switching schedule.

    Parameters
    ----------
    n_trials_per_segment : int, default 25
        Number of trials assigned to each segment in the sample schedule.

    Returns
    -------
    schedule : list of dict
        Explicit trial segments with keys `start_trial`, `end_trial`, and
        `strategy_name`.
    """
    if n_trials_per_segment <= 0:
        raise ValueError("n_trials_per_segment must be positive.")

    schedule = []
    start_trial = 0
    for strategy_name in DEFAULT_AGENT_SEQUENCE:
        end_trial = start_trial + n_trials_per_segment
        schedule.append(
            {
                "start_trial": start_trial,
                "end_trial": end_trial,
                "strategy_name": strategy_name,
            }
        )
        start_trial = end_trial
    return schedule


def make_sample_switch_params(
    n_trials_per_segment: int = DEFAULT_N_TRIALS_PER_SEGMENT,
) -> tuple[task_config.TaskParams, task_config.AgentParams]:
    """Create task and agent parameter sets for the sample switched run.

    Parameters
    ----------
    n_trials_per_segment : int, default 25
        Number of trials in each of the four default switching segments.

    Returns
    -------
    task_params : TaskParams
        Task parameters with `n_trials == 4 * n_trials_per_segment`.
    agent_params : AgentParams
        Shared agent parameter bundle for all sample strategies.
    """
    task_params = task_config.TaskParams(
        block_transition_style="markov",
        state_transition_prob=0.2,
        active_reward_probability=0.8,
        inactive_reward_probability=0.0,
        mean_correct_reward=1.0,
        mean_incorrect_reward=0.0,
        reward_std_dev=0.0,
        n_trials=4 * n_trials_per_segment,
    )
    agent_params = task_config.AgentParams(
        HMM_transition_prob=0.2,
        HMM_value_mode="bayesian_log_odds",
        HMM_log_odds_tanh_scale=1.2,
        HMM_reward_decay_lambda=0.35,
        relative_doubt_lambda=0.4,
        HMM_active_reward_probability=0.8,
        HMM_inactive_reward_probability=0.0,
        FQL_decay=0.7,
        logistic_alpha=0.0,
        action_temperature=1.0,
    )
    return task_params, agent_params


def build_sample_agents(
    agent_params: task_config.AgentParams,
    task_params: task_config.TaskParams,
    seed: int | None,
) -> dict[str, object]:
    """Construct the agents used in the default sample switching run.

    Parameters
    ----------
    agent_params : AgentParams
        Shared agent parameter bundle for all sample strategies.
    task_params : TaskParams
        Shared task parameter bundle for all sample strategies.
    seed : int or None
        Base seed used to derive per-agent random generators.

    Returns
    -------
    agents_by_name : dict[str, BehaviorAgent]
        Mapping from strategy name to instantiated agent.
    """
    if seed is None:
        agent_rngs = [None] * len(set(DEFAULT_AGENT_SEQUENCE))
    else:
        seed_sequence = np.random.SeedSequence(seed)
        agent_rngs = [
            np.random.default_rng(child_seed)
            for child_seed in seed_sequence.spawn(len(set(DEFAULT_AGENT_SEQUENCE)))
        ]

    strategy_names = ("HMM", "F-Qlearning", "HMM_reward_decay_relative_doubt")
    return {
        strategy_name: controller.select_agent(
            strategy_name,
            agent_params,
            task_params,
            rng=agent_rng,
        )
        for strategy_name, agent_rng in zip(strategy_names, agent_rngs)
    }


def build_default_figure_path(
    switching_mode: int,
    seed: int | None,
    figure_dir: Path | None = None,
) -> Path:
    """Build the default figure path for a sample switching run.

    Parameters
    ----------
    switching_mode : int
        Supported switching mode identifier (`1`, `2`, or `3`).
    seed : int or None
        Seed used for the run. Included in the filename for reproducibility.
    figure_dir : Path or None, default None
        Directory where figures should be written.

    Returns
    -------
    figure_path : Path
        Default PNG path for the saved sample-run figure.
    """
    if figure_dir is None:
        figure_dir = DEFAULT_FIGURE_DIR
    seed_label = "none" if seed is None else str(seed)
    return figure_dir / f"sample_switch_mode-{switching_mode}_seed-{seed_label}.png"


def run_sample_switch_demo(
    switching_mode: int,
    *,
    seed: int | None = 123,
    n_trials_per_segment: int = DEFAULT_N_TRIALS_PER_SEGMENT,
    save_figure: bool = True,
    save_run: bool = False,
    figure_dir: Path | None = None,
    theme: str = "light",
):
    """Run the default switched behavior-model demo and optionally save a figure.

    Parameters
    ----------
    switching_mode : int
        Supported switching mode identifier (`1`, `2`, or `3`).
    seed : int or None, default 123
        Seed controlling the task and agent random generators.
    n_trials_per_segment : int, default 25
        Number of trials assigned to each schedule segment.
    save_figure : bool, default True
        If True, save the default plot to `figure_dir`.
    save_run : bool, default False
        Reserved flag for future run saving. Currently only `False` is
        supported.
    figure_dir : Path or None, default None
        Directory where figures should be written if `save_figure` is True.
    theme : {"light", "dark"}, default "light"
        Plot theme used for the saved or returned figure.

    Returns
    -------
    artifacts : RunArtifacts
        Run dataframe and optional figure/save metadata.
    """
    if save_run:
        raise NotImplementedError("save_run is not implemented for sample switched runs yet.")

    switching._validate_switching_mode(switching_mode)
    task_params, agent_params = make_sample_switch_params(
        n_trials_per_segment=n_trials_per_segment
    )
    strategy_schedule = build_default_sample_switch_schedule(
        n_trials_per_segment=n_trials_per_segment
    )

    if seed is None:
        task_rng = None
        agent_seed = None
    else:
        task_seed, agent_seed = np.random.SeedSequence(seed).spawn(2)
        task_rng = np.random.default_rng(task_seed)
        agent_seed = int(np.random.default_rng(agent_seed).integers(0, 2**31 - 1))

    task = BaseMDP(task_params, rng=task_rng)
    agents_by_name = build_sample_agents(agent_params, task_params, seed=agent_seed)
    task, agents_by_name, performance_df = switching.run_switched_agent_session(
        task=task,
        agents_by_name=agents_by_name,
        strategy_schedule=strategy_schedule,
        switching_mode=switching_mode,
        task_params=task_params,
    )

    plot_path = None
    if save_figure:
        plot_path = build_default_figure_path(
            switching_mode=switching_mode,
            seed=seed,
            figure_dir=figure_dir,
        )

    figure, _ = controller.plot_run_dataframe(
        performance_df,
        f"sample-switch-mode-{switching_mode}",
        task_params,
        agent_params,
        plot_save_path=plot_path,
        show_plot=False,
        theme=theme,
    )
    return controller.RunArtifacts(
        performance_df=performance_df,
        run_path=None,
        plot_path=plot_path,
        figure=figure,
        params_path=None,
        seed=seed,
    )


def main() -> None:
    """Run a sample switched demo from the command line."""
    parser = argparse.ArgumentParser(description="Run a sample switched behavior-model demo.")
    parser.add_argument(
        "--mode",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Switching mode to run.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed for the run.")
    parser.add_argument(
        "--n-trials-per-segment",
        type=int,
        default=DEFAULT_N_TRIALS_PER_SEGMENT,
        help="Number of trials per strategy segment.",
    )
    parser.add_argument(
        "--no-save-figure",
        action="store_true",
        help="Do not save the output figure.",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="light",
        choices=["light", "dark"],
        help="Plot theme to use for the sample figure.",
    )
    args = parser.parse_args()

    artifacts = run_sample_switch_demo(
        switching_mode=args.mode,
        seed=args.seed,
        n_trials_per_segment=args.n_trials_per_segment,
        save_figure=not args.no_save_figure,
        theme=args.theme,
    )
    if artifacts.plot_path is not None:
        print(f"[***] Sample switching figure saved to: {artifacts.plot_path.resolve()}")


if __name__ == "__main__":
    main()
