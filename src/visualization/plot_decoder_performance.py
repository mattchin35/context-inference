import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from pathlib import Path
import time
from icecream import ic
import pickle as pkl
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import permutation_test_score, StratifiedKFold, StratifiedShuffleSplit


experiment_folder = Path('/home/matt/Documents/EXPERIMENTS/')
# experiment_folder = Path('C:/Users/mattc/EinsteinMed Dropbox/Matthew Chin/phd_data/remotework/EXPERIMENTS/')
behavior_data_path = experiment_folder / 'raw_behavior_data'
ephys_data_path = experiment_folder / 'processed_ephys_data'
processed_data_path = experiment_folder / 'processed_data'

current_mouse = 'CT010'
current_date_behavior = '2025-08-18'
current_date_ephys = ''.join(current_date_behavior.split('-'))  # remove dashes for ephys folder name
sess_timestamp = '124026'
sess_id = current_mouse + '_' + current_date_behavior
sess_id_full = current_mouse + '_' + current_date_behavior + '_' + sess_timestamp

behavior_session_path = behavior_data_path / sess_id_full
session_info_path = behavior_session_path / '{}_session_info.pkl'.format(sess_id_full)
output_path = processed_data_path / current_mouse
assert (output_path / 'figures').exists(), "Output path does not exist."

cv_results_path = processed_data_path / current_mouse / 'HPC_decoding_CV_results.csv'
shuffle_results_path = processed_data_path / current_mouse / 'HPC_decoding_shuffle_results.csv'
with open(cv_results_path, 'rb') as f:
    cv_results_df = pd.read_csv(f, sep=',')
with open(shuffle_results_path, 'rb') as f:
    shuffle_results_df = pd.read_csv(f, sep=',')

sessions = cv_results_df['sess_id'].unique()

before_ix = (shuffle_results_df['trial_type'] == 'correct') & (shuffle_results_df['bounds'] == '(-0.5, 0.0)')
after_ix = (shuffle_results_df['trial_type'] == 'correct') & (shuffle_results_df['bounds'] == '(0, 0.5)')

plt.figure(figsize=(8, 6))
before_data = shuffle_results_df['test_accuracy'][before_ix]
after_data = shuffle_results_df['test_accuracy'][after_ix]
for sess in sessions:
    sess_before_data = before_data[shuffle_results_df['sess_id'][before_ix] == sess]
    sess_after_data = after_data[shuffle_results_df['sess_id'][after_ix] == sess]
    plt.plot([0, 1], [sess_before_data, sess_after_data], color='gray', alpha=0.5)
plt.scatter(np.zeros(before_data.shape), before_data, color='C0', alpha=0.5)
plt.scatter(np.ones(after_data.shape)*1, after_data, color='C1', alpha=0.5)
plt.plot([0, 1], [before_data.mean(), after_data.mean()], color='k', linestyle='--', linewidth=2)
plt.xlim(-.5, 1.5)
plt.ylim(0, 1)
plt.xticks([0, 1], ['Before\nchoice', 'After\nchoice'])
plt.ylabel('Test accuracy')
plt.title('Decoder performance, correct trials')
# plt.show()
plt.tight_layout()
plt.savefig(output_path / 'figures' / 'shuffle_decoder_summary_correct_{}'.format(current_mouse), dpi=300)
plt.close()

before_ix = (shuffle_results_df['trial_type'] == 'incorrect') & (shuffle_results_df['bounds'] == '(-0.5, 0.0)')
after_ix = (shuffle_results_df['trial_type'] == 'incorrect') & (shuffle_results_df['bounds'] == '(0, 0.5)')

plt.figure(figsize=(8, 6))
before_data = shuffle_results_df['test_accuracy'][before_ix]
after_data = shuffle_results_df['test_accuracy'][after_ix]
for sess in sessions:
    sess_before_data = before_data[shuffle_results_df['sess_id'][before_ix] == sess]
    sess_after_data = after_data[shuffle_results_df['sess_id'][after_ix] == sess]
    plt.plot([0, 1], [sess_before_data, sess_after_data], color='gray', alpha=0.5)
plt.scatter(np.zeros(before_data.shape), before_data, color='C0', alpha=0.5)
plt.scatter(np.ones(after_data.shape)*1, after_data, color='C1', alpha=0.5)
plt.plot([0, 1], [before_data.mean(), after_data.mean()], color='k', linestyle='--', linewidth=2)
plt.xlim(-.5, 1.5)
plt.ylim(0, 1)
plt.xticks([0, 1], ['Before\nchoice', 'After\nchoice'])
plt.ylabel('Test accuracy')
plt.title('Decoder performance, incorrect trials')
# plt.show()
# plt.figsize=(4, 3)
plt.tight_layout()
plt.savefig(output_path / 'figures' / 'shuffle_decoder_summary_incorrect_{}'.format(current_mouse), dpi=300)
plt.close()

before_ix = (shuffle_results_df['trial_type'] == 'withheld') & (shuffle_results_df['bounds'] == '(-0.5, 0.0)')
after_ix = (shuffle_results_df['trial_type'] == 'withheld') & (shuffle_results_df['bounds'] == '(0, 0.5)')

plt.figure(figsize=(8, 6))
before_data = shuffle_results_df['test_accuracy'][before_ix]
after_data = shuffle_results_df['test_accuracy'][after_ix]
for sess in sessions:
    sess_before_data = before_data[shuffle_results_df['sess_id'][before_ix] == sess]
    sess_after_data = after_data[shuffle_results_df['sess_id'][after_ix] == sess]
    plt.plot([0, 1], [sess_before_data, sess_after_data], color='gray', alpha=0.5)
plt.scatter(np.zeros(before_data.shape), before_data, color='C0', alpha=0.5)
plt.scatter(np.ones(after_data.shape)*1, after_data, color='C1', alpha=0.5)
plt.plot([0, 1], [before_data.mean(), after_data.mean()], color='k', linestyle='--', linewidth=2)
plt.xlim(-.5, 1.5)
plt.ylim(0, 1)
plt.xticks([0, 1], ['Before\nchoice', 'After\nchoice'])
plt.ylabel('Test accuracy')
plt.title('Decoder performance, withheld trials')
# plt.show()
plt.tight_layout()
plt.savefig(output_path / 'figures' / 'shuffle_decoder_summary_withheld_{}'.format(current_mouse), dpi=300)
plt.close()


### CV results ###
before_ix = (cv_results_df['trial_type'] == 'correct') & (cv_results_df['bounds'] == '(-0.5, 0.0)')
after_ix = (cv_results_df['trial_type'] == 'correct') & ((cv_results_df['bounds'] == '(0, 0.5)') | (cv_results_df['bounds'] == '(0.0, 0.5)'))
plt.figure(figsize=(8, 6))
before_data = cv_results_df['cv_score'][before_ix]
after_data = cv_results_df['cv_score'][after_ix]
for sess in sessions:
    sess_before_data = before_data[cv_results_df['sess_id'][before_ix] == sess]
    sess_after_data = after_data[cv_results_df['sess_id'][after_ix] == sess]
    plt.plot([0, 1], [sess_before_data, sess_after_data], color='gray', alpha=0.5)
plt.scatter(np.zeros(before_data.shape), before_data, color='C0', alpha=0.5)
plt.scatter(np.ones(after_data.shape)*1, after_data, color='C1', alpha=0.5)
plt.plot([0, 1], [before_data.mean(), after_data.mean()], color='k', linestyle='--', linewidth=2)
plt.xlim(-.5, 1.5)
plt.ylim(0, 1)
plt.xticks([0, 1], ['Before\nchoice', 'After\nchoice'])
plt.ylabel('Test accuracy')
plt.title('Decoder performance, correct trials')
# plt.show()
plt.tight_layout()
plt.savefig(output_path / 'figures' / 'cv_decoder_summary_correct_{}'.format(current_mouse), dpi=300)
plt.close()

before_data = cv_results_df['cv_pvalue'][before_ix]
after_data = cv_results_df['cv_pvalue'][after_ix]
for sess in sessions:
    sess_before_data = before_data[cv_results_df['sess_id'][before_ix] == sess]
    sess_after_data = after_data[cv_results_df['sess_id'][after_ix] == sess]
    plt.plot([0, 1], [sess_before_data, sess_after_data], color='gray', alpha=0.5)
plt.scatter(np.zeros(before_data.shape), before_data, color='C0', alpha=0.5)
plt.scatter(np.ones(after_data.shape)*1, after_data, color='C1', alpha=0.5)
plt.plot([0, 1], [before_data.mean(), after_data.mean()], color='k', linestyle='--', linewidth=2)
plt.xlim(-.5, 1.5)
plt.ylim(0, 1)
plt.xticks([0, 1], ['Before\nchoice', 'After\nchoice'])
plt.ylabel('P value')
plt.title('Decoder performance, correct trials')
# plt.show()
plt.tight_layout()
plt.savefig(output_path / 'figures' / 'cv_decoder_pvalue_correct_{}'.format(current_mouse), dpi=300)
plt.close()

before_ix = (cv_results_df['trial_type'] == 'incorrect') & (cv_results_df['bounds'] == '(-0.5, 0.0)')
after_ix = (cv_results_df['trial_type'] == 'incorrect') & ((cv_results_df['bounds'] == '(0, 0.5)') | (cv_results_df['bounds'] == '(0.0, 0.5)'))
plt.figure(figsize=(8, 6))
before_data = cv_results_df['cv_score'][before_ix]
after_data = cv_results_df['cv_score'][after_ix]
for sess in sessions:
    sess_before_data = before_data[cv_results_df['sess_id'][before_ix] == sess]
    sess_after_data = after_data[cv_results_df['sess_id'][after_ix] == sess]
    plt.plot([0, 1], [sess_before_data, sess_after_data], color='gray', alpha=0.5)
plt.scatter(np.zeros(before_data.shape), before_data, color='C0', alpha=0.5)
plt.scatter(np.ones(after_data.shape)*1, after_data, color='C1', alpha=0.5)
plt.plot([0, 1], [before_data.mean(), after_data.mean()], color='k', linestyle='--', linewidth=2)
plt.xlim(-.5, 1.5)
plt.ylim(0, 1)
plt.xticks([0, 1], ['Before\nchoice', 'After\nchoice'])
plt.ylabel('Test accuracy')
plt.title('Decoder performance, incorrect trials')
plt.tight_layout()
plt.savefig(output_path / 'figures' / 'cv_decoder_summary_incorrect_{}'.format(current_mouse), dpi=300)
plt.close()

before_data = cv_results_df['cv_pvalue'][before_ix]
after_data = cv_results_df['cv_pvalue'][after_ix]
for sess in sessions:
    sess_before_data = before_data[cv_results_df['sess_id'][before_ix] == sess]
    sess_after_data = after_data[cv_results_df['sess_id'][after_ix] == sess]
    plt.plot([0, 1], [sess_before_data, sess_after_data], color='gray', alpha=0.5)
plt.scatter(np.zeros(before_data.shape), before_data, color='C0', alpha=0.5)
plt.scatter(np.ones(after_data.shape)*1, after_data, color='C1', alpha=0.5)
plt.plot([0, 1], [before_data.mean(), after_data.mean()], color='k', linestyle='--', linewidth=2)
plt.xlim(-.5, 1.5)
plt.ylim(0, 1)
plt.xticks([0, 1], ['Before\nchoice', 'After\nchoice'])
plt.ylabel('P value')
plt.title('Decoder performance, incorrect trials')
# plt.show()
plt.tight_layout()
plt.savefig(output_path / 'figures' / 'cv_decoder_pvalue_incorrect_{}'.format(current_mouse), dpi=300)
plt.close()


before_ix = (cv_results_df['trial_type'] == 'withheld') & (cv_results_df['bounds'] == '(-0.5, 0.0)')
after_ix = (cv_results_df['trial_type'] == 'withheld') & ((cv_results_df['bounds'] == '(0, 0.5)') | (cv_results_df['bounds'] == '(0.0, 0.5)'))
plt.figure(figsize=(8, 6))
before_data = cv_results_df['cv_score'][before_ix]
after_data = cv_results_df['cv_score'][after_ix]
for sess in sessions:
    sess_before_data = before_data[cv_results_df['sess_id'][before_ix] == sess]
    sess_after_data = after_data[cv_results_df['sess_id'][after_ix] == sess]
    plt.plot([0, 1], [sess_before_data, sess_after_data], color='gray', alpha=0.5)
plt.scatter(np.zeros(before_data.shape), before_data, color='C0', alpha=0.5)
plt.scatter(np.ones(after_data.shape)*1, after_data, color='C1', alpha=0.5)
plt.plot([0, 1], [before_data.mean(), after_data.mean()], color='k', linestyle='--', linewidth=2)
plt.xlim(-.5, 1.5)
plt.ylim(0, 1)
plt.xticks([0, 1], ['Before\nchoice', 'After\nchoice'])
plt.ylabel('Test accuracy')
plt.title('Decoder performance, withheld trials')
plt.tight_layout()
plt.savefig(output_path / 'figures' / 'cv_decoder_summary_withheld_{}'.format(current_mouse), dpi=300)
plt.close()


before_data = cv_results_df['cv_pvalue'][before_ix]
after_data = cv_results_df['cv_pvalue'][after_ix]
for sess in sessions:
    sess_before_data = before_data[cv_results_df['sess_id'][before_ix] == sess]
    sess_after_data = after_data[cv_results_df['sess_id'][after_ix] == sess]
    plt.plot([0, 1], [sess_before_data, sess_after_data], color='gray', alpha=0.5)
plt.scatter(np.zeros(before_data.shape), before_data, color='C0', alpha=0.5)
plt.scatter(np.ones(after_data.shape)*1, after_data, color='C1', alpha=0.5)
plt.plot([0, 1], [before_data.mean(), after_data.mean()], color='k', linestyle='--', linewidth=2)
plt.xlim(-.5, 1.5)
plt.ylim(0, 1)
plt.xticks([0, 1], ['Before\nchoice', 'After\nchoice'])
plt.ylabel('P value')
plt.title('Decoder performance, withheld trials')
# plt.show()
plt.tight_layout()
plt.savefig(output_path / 'figures' / 'cv_decoder_pvalue_withheld_{}'.format(current_mouse), dpi=300)
plt.close()

