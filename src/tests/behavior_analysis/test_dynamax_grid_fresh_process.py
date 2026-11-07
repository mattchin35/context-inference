"""Real fresh-process summary rendering and explicit full-block coordinates."""

import os
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def test_actual_multisession_grid_after_main_import_is_light_and_ssm_free(tmp_path):
    """Real metric/grid renderers ignore an invalid legacy pickle after dark imports."""
    script = r'''
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import src.main
from src.behavior_analysis import performance_plots as plots
root = Path(sys.argv[1])
(root / 'joint_block_statedict.pkl').write_bytes(b'invalid legacy pickle must not load')
date = '2026-01-01'
blocks = pd.DataFrame({'block_ix': [4, 8, 12, 16], 'prev_n_correct': [1, 2, 3, 4],
    'prev_n_rewarded': [1., 2., 3., 4.], 'trials_to_correct': [2., 3., 4., 5.],
    'source_session_id': ['same_day_a', 'same_day_a', 'same_day_b', 'same_day_b'],
    'exemplar_hmm_smoothed_state': [0, 0, 1, 1],
    'exemplar_hmm_smoothed_prob_A': [.9, .8, .3, .1],
    'exemplar_hmm_smoothed_prob_B': [.1, .2, .7, .9]})
side = pd.DataFrame({'date': [date, date], 'session_id': ['s', 's'],
    'rewarded_side': ['left', 'right'], 'n_blocks': [2, 2], 'n_valid_blocks': [2, 2],
    'n_no_correct_blocks': [0, 0], 'completion_fraction': [1., 1.],
    'trials_to_correct_q1': [2., 3.], 'trials_to_correct_median': [3., 4.],
    'trials_to_correct_q3': [4., 5.]})
side_points = pd.DataFrame({'date': [date], 'session_id': ['s'], 'rewarded_side': ['left'],
    'block_ix': [0], 'block_type': ['left_cued'], 'trials_to_correct_numeric': [3.], 'no_correct_choice': [False]})
correct = pd.DataFrame({'date': [date], 'correct_after_first_group': ['overall'],
    'percent_correct_after_first_q1': [.5], 'percent_correct_after_first_median': [.7],
    'percent_correct_after_first_q3': [.9]})
correct_points = pd.DataFrame({'date': [date], 'correct_after_first_group': ['overall'],
    'percent_correct_after_first_numeric': [.7], 'no_correct_choice': [False]})
switch = pd.DataFrame({'date': [date], 'switch_group': ['overall'],
    'n_switches_q1': [1.], 'n_switches_median': [2.], 'n_switches_q3': [3.]})
switch_points = pd.DataFrame({'date': [date], 'switch_group': ['overall'], 'n_switches': [2]})
overall = pd.DataFrame({'date': [date], 'history_ideal_oracle_accuracy': [.8],
    'history_ideal_mouse_agreement': [.7], 'fixed_replay_ideal_oracle_accuracy_q1': [.6],
    'fixed_replay_ideal_oracle_accuracy_median': [.7], 'fixed_replay_ideal_oracle_accuracy_q3': [.8]})
kwargs = dict(block_performance=blocks, side_trials_to_correct_summary=side,
    side_trials_to_correct_block_points=side_points, correct_after_first_summary=correct,
    correct_after_first_block_points=correct_points, block_switch_summary=switch,
    block_switch_block_points=switch_points, overall_df=overall, plot_path=root,
    figure_id='joint', processed_data_path=root, exemplar_model_names=['A', 'B'],
    sliding_regression_window_size=2, sliding_regression_step_size=1)
first = plots.plot_multisession_summary_grid(**kwargs)
second = plots.plot_multisession_summary_grid(**kwargs)
assert first != second and first.parent == root and first.exists()
pixels = mpimg.imread(first)
assert np.allclose(pixels[0, 0, :3], 1.)
assert pixels.shape[2] != 4 or pixels[0, 0, 3] == 1.
assert 'ssm' not in sys.modules
assert 'src.behavior_analysis.block_state_space_modeling' not in sys.modules
'''
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_summary_panel_labels_source_ids_and_probability_units():
    """Same-date sources use full-row boundaries and posterior-only probability scale."""
    from src.behavior_analysis import performance_plots as plots

    blocks = pd.DataFrame({
        "source_session_id": ["same_day_a", "same_day_a", "same_day_b", "same_day_b"],
        "exemplar_hmm_smoothed_prob_A": [.9, np.nan, .3, .1],
        "exemplar_hmm_smoothed_prob_B": [.1, np.nan, .7, .9],
    })
    fig, axes = plt.subplots(1, 2)
    try:
        plots._plot_exemplar_summary_grid_panels(axes[0], axes[1], blocks, ["A", "B"], 2, 1)
        assert axes[0].get_ylim() == (0., 1.)
        assert "probability" in axes[0].get_ylabel().lower()
        assert any("same_day_b" in text.get_text() for text in axes[0].texts)
        assert list(axes[0].lines[0].get_xdata()) == [0, 1, 2, 3]
        assert any("regression" in text.get_text().lower() for text in axes[1].texts)
    finally:
        plt.close(fig)
