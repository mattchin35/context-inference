"""Sample switched-run entry point for visualizing behavior-model transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.behavior_modeling import controller
from src.behavior_modeling.agent_switching import switching
from src.behavior_modeling.parameters import task_config
from src.behavior_modeling.task.context_task import BaseMDP


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "reports/figures/model_behavior/switching"
DEFAULT_RUN_DATA_DIR = PROJECT_ROOT / "data/processed/switching_agents"
DEFAULT_N_TRIALS_PER_SEGMENT = 25
DEFAULT_AGENT_SEQUENCE = (
    "HMM",
    "F-Qlearning",
    "HMM",
    "HMM_reward_decay_relative_doubt",
)
VALIDATION_AGENT_SEQUENCE = (
    "HMM",
    "F-Qlearning",
    "HMM_reward_decay_relative_doubt",
    "F-Qlearning",
)
VALIDATION_RECOMMENDED_TRIALS_PER_SEGMENT = 100


def _build_switch_schedule(
    agent_sequence: tuple[str, ...],
    n_trials_per_segment: int,
) -> list[dict[str, int | str]]:
    """Build an explicit switching schedule from an agent sequence.

    Parameters
    ----------
    agent_sequence : tuple[str, ...]
        Ordered strategy names used for consecutive schedule segments.
    n_trials_per_segment : int
        Number of trials assigned to each segment. Must be positive.

    Returns
    -------
    list of dict
        Schedule segments with `start_trial`, `end_trial`, and `strategy_name`.
    """
    if n_trials_per_segment <= 0:
        raise ValueError("n_trials_per_segment must be positive.")

    schedule = []
    start_trial = 0
    for strategy_name in agent_sequence:
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
    return _build_switch_schedule(DEFAULT_AGENT_SEQUENCE, n_trials_per_segment)


def build_validation_sample_switch_schedule(
    n_trials_per_segment: int = VALIDATION_RECOMMENDED_TRIALS_PER_SEGMENT,
) -> list[dict[str, int | str]]:
    """Build the validation-oriented switching schedule.

    Parameters
    ----------
    n_trials_per_segment : int, default 100
        Number of trials assigned to each validation segment. Validation runs
        recommend `100` trials per segment so downstream LM-HMM and GLM-HMM
        analyses have enough data to behave sensibly.

    Returns
    -------
    list of dict
        Validation schedule using the fixed recommended agent sequence:
        `HMM -> F-Qlearning -> HMM_reward_decay_relative_doubt -> F-Qlearning`.
    """
    return _build_switch_schedule(VALIDATION_AGENT_SEQUENCE, n_trials_per_segment)


def make_sample_switch_params(
    n_trials_per_segment: int = DEFAULT_N_TRIALS_PER_SEGMENT,
    *,
    block_transition_style: str = "markov",
    state_transition_prob: float = 0.2,
    default_fixed_block_length: int = 15,
    fixed_block_length_variation: int = 0,
    success_trials_to_block_transition: float = np.inf,
    p_cue: float = 0.25,
) -> tuple[task_config.TaskParams, task_config.AgentParams]:
    """Create task and agent parameter sets for the sample switched run.

    Parameters
    ----------
    n_trials_per_segment : int, default 25
        Number of trials in each of the four default switching segments.
    block_transition_style : str, default "markov"
        Task block-transition mode. Validation runs typically use `"fixed"`.
    state_transition_prob : float, default 0.2
        Task switch probability for `markov` and `success_trigger` modes.
    default_fixed_block_length : int, default 15
        Fixed block length used when `block_transition_style == "fixed"`.
    fixed_block_length_variation : int, default 0
        Allowed fixed-block-length variation around the default.
    success_trials_to_block_transition : float, default np.inf
        Number of correct trials needed to switch when using `n_correct`.
    p_cue : float, default 0.25
        Probability that a trial carries a cue. Validation runs recommend
        `0.5`, and `100` trials per segment overall, for more analysis-friendly
        sessions.

    Returns
    -------
    task_params : TaskParams
        Task parameters with `n_trials == 4 * n_trials_per_segment`.
    agent_params : AgentParams
        Shared agent parameter bundle for all sample strategies.
    """
    task_params = task_config.TaskParams(
        block_transition_style=block_transition_style,
        p_cue=p_cue,
        state_transition_prob=state_transition_prob,
        active_reward_probability=0.8,
        inactive_reward_probability=0.0,
        mean_correct_reward=1.0,
        mean_incorrect_reward=0.0,
        reward_std_dev=0.0,
        n_trials=4 * n_trials_per_segment,
        default_fixed_block_length=default_fixed_block_length,
        fixed_block_length_variation=fixed_block_length_variation,
        success_trials_to_block_transition=success_trials_to_block_transition,
    )
    # 'BaseMDP' still reads the older fixed-block attribute names. Keep the
    # compatibility shim local to this entry point rather than widening this
    # preset change into task code.
    task_params.default_block_length = task_params.default_fixed_block_length
    task_params.block_length_variation = task_params.fixed_block_length_variation
    agent_params = task_config.AgentParams(
        HMM_transition_prob=0.1,
        HMM_value_mode="bayesian_log_odds",
        HMM_log_odds_tanh_scale=1.2,
        HMM_reward_decay_lambda=0.2,
        relative_doubt_lambda=0.2,
        HMM_active_reward_probability=0.8,
        HMM_inactive_reward_probability=0.0,
        FQL_decay=0.4,
        logistic_alpha=0.0,
        action_temperature=1.0,
    )
    return task_params, agent_params


def apply_validation_preset(
    *,
    n_trials_per_segment: int | None = None,
    block_transition_style: str | None = None,
    state_transition_prob: float | None = None,
    default_fixed_block_length: int | None = None,
    fixed_block_length_variation: int | None = None,
    success_trials_to_block_transition: float | None = None,
    p_cue: float | None = None,
) -> dict[str, int | float | str]:
    """Return validation-oriented task settings with user-overridable values.

    Validation runs recommend `100` trials per segment as a starting point.
    Explicitly supplied values override the preset.
    """
    return {
        "n_trials_per_segment": (
            VALIDATION_RECOMMENDED_TRIALS_PER_SEGMENT
            if n_trials_per_segment is None
            else n_trials_per_segment
        ),
        "block_transition_style": "fixed" if block_transition_style is None else block_transition_style,
        "state_transition_prob": 0.2 if state_transition_prob is None else state_transition_prob,
        "default_fixed_block_length": 15 if default_fixed_block_length is None else default_fixed_block_length,
        "fixed_block_length_variation": 0 if fixed_block_length_variation is None else fixed_block_length_variation,
        "success_trials_to_block_transition": (
            np.inf if success_trials_to_block_transition is None else success_trials_to_block_transition
        ),
        "p_cue": 0.5 if p_cue is None else p_cue,
    }


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


def build_default_run_root(
    switching_mode: int,
    run_data_dir: Path | None = None,
) -> Path:
    """Build the output root for saved switched runs.

    Parameters
    ----------
    switching_mode : int
        Supported switching mode identifier (`1`, `2`, or `3`).
    run_data_dir : Path or None, default None
        Directory containing all saved switched-run folders.

    Returns
    -------
    Path
        Directory of the form `<run_data_dir>/sample_switch_mode_<mode>`.
    """
    if run_data_dir is None:
        run_data_dir = DEFAULT_RUN_DATA_DIR
    return run_data_dir / f"sample_switch_mode_{switching_mode}"


def build_default_run_stem(
    switching_mode: int,
    task_params: task_config.TaskParams,
    agent_params: task_config.AgentParams,
    session_timestamp: str | None = None,
) -> str:
    """Build the filename stem for a saved sample switched run.

    Parameters
    ----------
    switching_mode : int
        Supported switching mode identifier (`1`, `2`, or `3`).
    task_params : TaskParams
        Task parameters used for the run.
    agent_params : AgentParams
        Shared agent parameters used for all strategies in the run.

    Returns
    -------
    str
        Timestamped filename stem compatible with the sample-agent CSV+JSON
        save format.
    """
    if session_timestamp is None:
        session_timestamp = controller.get_session_timestamp()
    param_summary = controller.format_agent_parameter_summary(
        f"sample_switch_mode_{switching_mode}",
        task_params,
        agent_params,
    )
    return f"sample_switch_mode_{switching_mode}_{session_timestamp}_{param_summary}"


def summarize_strategy_schedule(
    strategy_schedule: list[dict[str, int | str]],
) -> dict[str, list[int]]:
    """Create a compact human-readable schedule summary.

    Parameters
    ----------
    strategy_schedule : list of dict
        Ordered schedule segments with inclusive `start_trial` and exclusive
        `end_trial` bounds.

    Returns
    -------
    dict[str, list[int]]
        Mapping like `model0: [start, stop]`, where `stop` is inclusive.
    """
    schedule_by_model: dict[str, list[int]] = {}
    for idx, segment in enumerate(strategy_schedule):
        schedule_by_model[f"model{idx}"] = [
            int(segment["start_trial"]),
            int(segment["end_trial"]) - 1,
        ]
    return schedule_by_model


def save_sample_switch_run(
    performance_df,
    task_params: task_config.TaskParams,
    agent_params: task_config.AgentParams,
    switching_mode: int,
    strategy_schedule: list[dict[str, int | str]],
    seed: int | None,
    run_data_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Save a sample switched run as CSV + JSON metadata.

    Parameters
    ----------
    performance_df : pd.DataFrame
        Trial-wise switched-run dataframe. Must include `active_strategy`.
    task_params : TaskParams
        Task parameters used for the run.
    agent_params : AgentParams
        Shared agent parameters for all switched agents.
    switching_mode : int
        Supported switching mode identifier (`1`, `2`, or `3`).
    strategy_schedule : list of dict
        Ordered schedule segments with `start_trial`, `end_trial`, and
        `strategy_name`.
    seed : int or None
        Seed used for the run.
    run_data_dir : Path or None, default None
        Root directory for switched-run outputs.

    Returns
    -------
    tuple[Path, Path]
        Saved CSV path and saved JSON metadata path.
    """
    if "active_strategy" not in performance_df.columns:
        raise ValueError("Switched-run dataframe must contain an 'active_strategy' column.")

    run_root = build_default_run_root(switching_mode=switching_mode, run_data_dir=run_data_dir)
    session_timestamp = controller.get_session_timestamp()
    run_stem = build_default_run_stem(
        switching_mode=switching_mode,
        task_params=task_params,
        agent_params=agent_params,
        session_timestamp=session_timestamp,
    )
    output_dir = run_root / run_stem
    output_dir.mkdir(parents=True, exist_ok=False)

    run_path = output_dir / f"{run_stem}.csv"
    params_path = output_dir / f"{run_stem}_params.json"
    performance_df.to_csv(run_path, index=False)

    params_payload = {
        "agent_name": f"sample_switch_mode_{switching_mode}",
        "model_type_label": f"sample_switch_mode_{switching_mode}",
        "session_timestamp": session_timestamp,
        "run_seed": seed,
        "task_params": vars(task_params),
        "agent_params": vars(agent_params),
        "dataframe_columns": list(performance_df.columns),
        "switching_mode": switching_mode,
        "agents": sorted({str(segment["strategy_name"]) for segment in strategy_schedule}),
        "strategy_schedule": strategy_schedule,
        "schedule_by_model": summarize_strategy_schedule(strategy_schedule),
    }
    with params_path.open("w") as f:
        json.dump(params_payload, f, indent=2)

    return run_path, params_path


def run_sample_switch_demo(
    switching_mode: int,
    *,
    seed: int | None = 123,
    n_trials_per_segment: int = DEFAULT_N_TRIALS_PER_SEGMENT,
    validation_run: bool = False,
    block_transition_style: str = "markov",
    state_transition_prob: float = 0.2,
    default_fixed_block_length: int = 15,
    fixed_block_length_variation: int = 0,
    success_trials_to_block_transition: float = np.inf,
    p_cue: float = 0.25,
    save_figure: bool = True,
    save_run: bool = False,
    figure_dir: Path | None = None,
    run_data_dir: Path | None = None,
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
    validation_run : bool, default False
        If True, apply a validation-oriented preset. Validation runs recommend
        `100` trials per segment, use fixed-length blocks, and swap in a more
        separable agent sequence for downstream analysis. Explicit knob values
        supplied to this function override the preset.
    block_transition_style : str, default "markov"
        Task block-transition mode. Exposed as a free-form override knob.
    state_transition_prob : float, default 0.2
        Task switch probability for `markov` and `success_trigger` modes.
    default_fixed_block_length : int, default 15
        Fixed block length used when `block_transition_style == "fixed"`.
    fixed_block_length_variation : int, default 0
        Variation around `default_fixed_block_length` in fixed mode.
    success_trials_to_block_transition : float, default np.inf
        Correct-trial threshold for `n_correct` transition mode.
    p_cue : float, default 0.25
        Cue probability. Validation runs recommend `0.5`.
    save_figure : bool, default True
        If True, save the default plot to `figure_dir`.
    save_run : bool, default False
        If True, save the switched run to CSV plus JSON metadata.
    figure_dir : Path or None, default None
        Directory where figures should be written if `save_figure` is True.
    run_data_dir : Path or None, default None
        Root directory where switched-run CSV + JSON outputs should be written.
    theme : {"light", "dark"}, default "light"
        Plot theme used for the saved or returned figure.

    Returns
    -------
    artifacts : RunArtifacts
        Run dataframe and optional figure/save metadata.
    """
    switching._validate_switching_mode(switching_mode)
    resolved_n_trials_per_segment = n_trials_per_segment
    resolved_block_transition_style = block_transition_style
    resolved_state_transition_prob = state_transition_prob
    resolved_default_fixed_block_length = default_fixed_block_length
    resolved_fixed_block_length_variation = fixed_block_length_variation
    resolved_success_trials_to_block_transition = success_trials_to_block_transition
    resolved_p_cue = p_cue

    if validation_run:
        preset = apply_validation_preset(
            n_trials_per_segment=n_trials_per_segment if n_trials_per_segment != DEFAULT_N_TRIALS_PER_SEGMENT else None,
            block_transition_style=block_transition_style if block_transition_style != "markov" else None,
            state_transition_prob=state_transition_prob if state_transition_prob != 0.2 else None,
            default_fixed_block_length=(
                default_fixed_block_length if default_fixed_block_length != 15 else None
            ),
            fixed_block_length_variation=(
                fixed_block_length_variation if fixed_block_length_variation != 0 else None
            ),
            success_trials_to_block_transition=(
                success_trials_to_block_transition
                if success_trials_to_block_transition != np.inf
                else None
            ),
            p_cue=p_cue if p_cue != 0.25 else None,
        )
        resolved_n_trials_per_segment = int(preset["n_trials_per_segment"])
        resolved_block_transition_style = str(preset["block_transition_style"])
        resolved_state_transition_prob = float(preset["state_transition_prob"])
        resolved_default_fixed_block_length = int(preset["default_fixed_block_length"])
        resolved_fixed_block_length_variation = int(preset["fixed_block_length_variation"])
        resolved_success_trials_to_block_transition = float(
            preset["success_trials_to_block_transition"]
        )
        resolved_p_cue = float(preset["p_cue"])

    task_params, agent_params = make_sample_switch_params(
        n_trials_per_segment=resolved_n_trials_per_segment,
        block_transition_style=resolved_block_transition_style,
        state_transition_prob=resolved_state_transition_prob,
        default_fixed_block_length=resolved_default_fixed_block_length,
        fixed_block_length_variation=resolved_fixed_block_length_variation,
        success_trials_to_block_transition=resolved_success_trials_to_block_transition,
        p_cue=resolved_p_cue,
    )
    strategy_schedule = (
        build_validation_sample_switch_schedule(
            n_trials_per_segment=resolved_n_trials_per_segment
        )
        if validation_run
        else build_default_sample_switch_schedule(
            n_trials_per_segment=resolved_n_trials_per_segment
        )
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

    run_path = None
    params_path = None
    if save_run:
        run_path, params_path = save_sample_switch_run(
            performance_df=performance_df,
            task_params=task_params,
            agent_params=agent_params,
            switching_mode=switching_mode,
            strategy_schedule=strategy_schedule,
            seed=seed,
            run_data_dir=run_data_dir,
        )

    return controller.RunArtifacts(
        performance_df=performance_df,
        run_path=run_path,
        plot_path=plot_path,
        figure=figure,
        params_path=params_path,
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
        "--validation-run",
        action="store_true",
        help="Use the validation preset. Validation runs recommend 100 trials per segment.",
    )
    parser.add_argument(
        "--block-transition-style",
        type=str,
        default="markov",
        choices=["markov", "success_trigger", "n_correct", "fixed"],
        help="Task block-transition style. Validation runs typically use fixed blocks.",
    )
    parser.add_argument(
        "--state-transition-prob",
        type=float,
        default=0.2,
        help="Task switch probability for markov and success_trigger modes.",
    )
    parser.add_argument(
        "--default-fixed-block-length",
        type=int,
        default=15,
        help="Fixed block length when using fixed transitions.",
    )
    parser.add_argument(
        "--fixed-block-length-variation",
        type=int,
        default=0,
        help="Variation around the default fixed block length.",
    )
    parser.add_argument(
        "--success-trials-to-block-transition",
        type=float,
        default=float("inf"),
        help="Correct-trial threshold for n_correct transition mode.",
    )
    parser.add_argument(
        "--p-cue",
        type=float,
        default=0.25,
        help="Cue probability. Validation runs recommend 0.5.",
    )
    parser.add_argument(
        "--no-save-figure",
        action="store_true",
        help="Do not save the output figure.",
    )
    parser.add_argument(
        "--save-run",
        action="store_true",
        help="Save the switched run as CSV plus JSON metadata.",
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
        validation_run=args.validation_run,
        block_transition_style=args.block_transition_style,
        state_transition_prob=args.state_transition_prob,
        default_fixed_block_length=args.default_fixed_block_length,
        fixed_block_length_variation=args.fixed_block_length_variation,
        success_trials_to_block_transition=args.success_trials_to_block_transition,
        p_cue=args.p_cue,
        save_figure=not args.no_save_figure,
        save_run=args.save_run,
        theme=args.theme,
    )
    if artifacts.plot_path is not None:
        print(f"[***] Sample switching figure saved to: {artifacts.plot_path.resolve()}")
    if artifacts.run_path is not None:
        print(f"[***] Sample switching run saved to: {artifacts.run_path.resolve()}")
    if artifacts.params_path is not None:
        print(f"[***] Sample switching params saved to: {artifacts.params_path.resolve()}")


if __name__ == "__main__":
    main()
