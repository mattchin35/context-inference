"""Integration contracts for expectant-switching features and agents."""

import sys
from types import ModuleType

import numpy as np
import numpy.testing as npt
import pandas as pd

from src.behavior_analysis import gather_trial_features
from src.behavior_analysis import performance_plots
from src.behavior_analysis import session_analysis

# The registry tested here is import-time data and does not require the
# optional GLM-HMM runtime dependency.
sys.modules.setdefault("ssm", ModuleType("ssm"))
ssm_plots_stub = ModuleType("ssm.plots")
ssm_plots_stub.gradient_cmap = lambda *_args, **_kwargs: None
ssm_plots_stub.white_to_color_cmap = lambda *_args, **_kwargs: None
sys.modules.setdefault("ssm.plots", ssm_plots_stub)
from src.behavior_analysis import trial_state_space_modeling
from src.behavior_modeling import controller
from src.behavior_modeling.agents import agents
from src.behavior_modeling.expectant_switching import (
    ExpectancyPersistenceDoubtParams,
    SimpleProbePersistenceParams,
    replay_expectant_switching,
    ExpectantSwitchingFeatureConfig,
)
from src.behavior_modeling.parameters import task_config


EXPECTED_FEATURE_COLUMNS = {
    "previous_choice",
    "previous_reward",
    "reward_triggered_probe",
    "expectancy_confirmed_side",
    "expectancy_reward_count",
    "expectancy_strength",
    "expectant_switch",
    "doubt_raw_value",
    "doubt_choice_signal",
    "simple_probe_persistence_drive",
    "simple_probe_persistence_value",
    "simple_probe_persistence_prob_left",
    "expectancy_persistence_doubt_drive",
    "expectancy_persistence_doubt_value",
    "expectancy_persistence_doubt_prob_left",
}


def test_analysis_params_expose_adjustable_documented_defaults():
    params = gather_trial_features.TaskParams()

    assert params.expectancy_threshold == 3.0
    assert params.expectancy_scale == 1.0
    assert params.simple_persistence_weight == 1.0
    assert params.simple_probe_weight == 1.0
    assert params.full_persistence_weight == 1.0
    assert params.full_expectancy_weight == 1.0
    assert params.full_doubt_weight == 1.0


def test_trial_feature_adapter_adds_aligned_outputs_and_skips_manual_reward():
    frame = pd.DataFrame(
        {
            "action": [1, "no_choice", 0, 1, 0],
            "reward": [1.0, 0.0, 1.0, 0.0, 0.0],
            "experimenter_reward_given": [0, 0, 1, 0, 0],
        }
    )

    result = gather_trial_features.collect_expectant_switching_features(
        frame,
        gather_trial_features.TaskParams(),
    )

    assert EXPECTED_FEATURE_COLUMNS.issubset(result.columns)
    assert result.loc[0, "previous_choice"] == 0.0
    assert result.loc[3, "previous_choice"] == 1.0
    assert result.loc[3, "expectancy_reward_count"] == 1.0
    assert result.loc[4, "previous_choice"] == 1.0
    assert result.loc[4, "previous_reward"] == 0.0
    for invalid_index in (1, 2):
        assert pd.isna(result.loc[invalid_index, "simple_probe_persistence_value"])


def test_signed_expectant_columns_are_side_equivalent_but_counts_are_not():
    directional = gather_trial_features.LEFT_RIGHT_VALUE_COLUMNS_FOR_SIDE_EQUIVALENCE

    for column in (
        "previous_choice",
        "reward_triggered_probe",
        "expectant_switch",
        "doubt_raw_value",
        "doubt_choice_signal",
        "simple_probe_persistence_drive",
        "simple_probe_persistence_value",
        "expectancy_persistence_doubt_drive",
        "expectancy_persistence_doubt_value",
    ):
        assert column in directional
    for column in (
        "previous_reward",
        "expectancy_confirmed_side",
        "expectancy_reward_count",
        "expectancy_strength",
        "simple_probe_persistence_prob_left",
        "expectancy_persistence_doubt_prob_left",
    ):
        assert column not in directional


def test_agreement_plot_and_glm_registries_include_final_values_only():
    assert session_analysis.DEFAULT_AGENT_MOUSE_AGREEMENT_VALUE_COLUMNS[
        "simple_probe_persistence_mouse_agreement"
    ] == "simple_probe_persistence_value"
    assert session_analysis.DEFAULT_AGENT_MOUSE_AGREEMENT_VALUE_COLUMNS[
        "expectancy_persistence_doubt_mouse_agreement"
    ] == "expectancy_persistence_doubt_value"
    assert performance_plots.DEFAULT_AGENT_MOUSE_AGREEMENT_COLUMNS[
        "Probe+P"
    ] == "simple_probe_persistence_mouse_agreement"
    assert performance_plots.DEFAULT_AGENT_MOUSE_AGREEMENT_COLUMNS[
        "Expect+D+P"
    ] == "expectancy_persistence_doubt_mouse_agreement"
    assert "Probe+P" in performance_plots.AGENT_MOUSE_AGREEMENT_COLORS
    assert "Expect+D+P" in performance_plots.AGENT_MOUSE_AGREEMENT_COLORS

    predictor_labels = trial_state_space_modeling.TRIAL_GLM_PREDICTOR_LABELS
    assert "simple_probe_persistence_value" in predictor_labels
    assert "expectancy_persistence_doubt_value" in predictor_labels
    assert "expectancy_reward_count" not in predictor_labels
    assert "simple_probe_persistence_value" not in (
        trial_state_space_modeling.DEFAULT_TRIAL_GLM_PREDICTOR_COLUMNS
    )


def make_agent_params() -> tuple[task_config.AgentParams, task_config.TaskParams]:
    """Return deterministic task and documented exemplar parameters."""
    agent_params = task_config.AgentParams(
        greedy_action_selection=False,
        expectancy_threshold=3.0,
        expectancy_scale=1.0,
        simple_persistence_weight=1.0,
        simple_probe_weight=1.0,
        full_persistence_weight=1.0,
        full_expectancy_weight=1.0,
        full_doubt_weight=1.0,
        relative_doubt_lambda=0.5,
    )
    task_params = task_config.TaskParams(
        mean_correct_reward=1.0,
        mean_incorrect_reward=0.0,
        reward_std_dev=0.0,
    )
    return agent_params, task_params


def test_controller_selects_both_agents_with_injected_rng():
    agent_params, task_params = make_agent_params()
    rng = np.random.default_rng(42)

    simple = controller.select_agent(
        "simple_probe_persistence", agent_params, task_params, rng=rng
    )
    full = controller.select_agent(
        "expectancy_persistence_doubt", agent_params, task_params, rng=rng
    )

    assert isinstance(simple, agents.SimpleProbePersistenceAgent)
    assert isinstance(full, agents.ExpectancyPersistenceDoubtAgent)
    assert simple.rng is rng
    assert full.rng is rng
    assert simple.model_type == "simple_probe_persistence"
    assert full.model_type == "expectancy_persistence_doubt"


def test_online_agents_match_batch_replay_pretrial_values():
    actions = np.array([1, 1, 0, 0, 1], dtype=int)
    rewards = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
    valid = np.ones(actions.shape, dtype=bool)
    agent_params, task_params = make_agent_params()
    config = ExpectantSwitchingFeatureConfig(
        simple=SimpleProbePersistenceParams(),
        full=ExpectancyPersistenceDoubtParams(doubt_lambda=0.5),
    )
    replay = replay_expectant_switching(actions, rewards, valid, config)
    simple = agents.SimpleProbePersistenceAgent(
        agent_params, task_params, rng=np.random.default_rng(1)
    )
    full = agents.ExpectancyPersistenceDoubtAgent(
        agent_params, task_params, rng=np.random.default_rng(2)
    )

    simple_values = []
    full_values = []
    full_probabilities = []
    for action, reward in zip(actions, rewards):
        simple_values.append(simple.value)
        full_values.append(full.value)
        full_probabilities.append(full.action_dist[1])
        simple.update_params(action, reward)
        full.update_params(action, reward)

    npt.assert_allclose(simple_values, replay["simple_probe_persistence_value"])
    npt.assert_allclose(full_values, replay["expectancy_persistence_doubt_value"])
    npt.assert_allclose(
        full_probabilities,
        replay["expectancy_persistence_doubt_prob_left"],
    )


def test_greedy_tie_repeats_previous_model_choice():
    agent_params, task_params = make_agent_params()
    agent_params.greedy_action_selection = True
    agent_params.greedy_epsilon = 0.0
    agent = agents.SimpleProbePersistenceAgent(
        agent_params, task_params, rng=np.random.default_rng(5)
    )

    initial_action, _ = agent.choose_action(stimulus=-1)
    assert initial_action == 0

    agent.update_params(action=1, reward=1.0)
    assert agent.value == 0.0
    tied_action, _ = agent.choose_action(stimulus=-1)
    assert tied_action == 1


def test_controller_performance_includes_expectant_agent_diagnostics():
    agent_params, task_params = make_agent_params()
    task_params.n_trials = 4
    agent = controller.select_agent(
        "expectancy_persistence_doubt",
        agent_params,
        task_params,
        rng=np.random.default_rng(9),
    )
    task = controller.BaseMDP(task_params, rng=np.random.default_rng(10))

    _, _, performance = controller.run_agent_session(task, agent, task_params)

    for column in (
        "agent_previous_choice",
        "agent_previous_reward",
        "agent_reward_triggered_probe",
        "agent_expectancy_reward_count",
        "agent_expectancy_strength",
        "agent_expectant_switch",
        "agent_doubt_raw_value",
        "agent_doubt_choice_signal",
        "agent_decision_drive",
        "agent_prob_left",
    ):
        assert column in performance.columns
        assert performance[column].notna().any()
