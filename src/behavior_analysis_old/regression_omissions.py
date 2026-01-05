import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from formulaic import model_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import Any, Iterable
import src.behavior_analysis.context_switch_analysis as vdp


def session_stats(df: pd.DataFrame, key: Any, plot=False):
    stats_df = pd.DataFrame({
        'y': df[key],
        'a': df['consecutive_rewards'],
    })

    y, X = model_matrix("y ~ a", stats_df)
    model = sm.OLS(y, X)
    results = model.fit()
    # print(results.summary())
    # print("Parameters", results.params)
    # print("R-squared", results.rsquared)
    # print("Std Errors", results.bse)

    if plot:
        pred_ols = results.get_prediction()
        iv_l = pred_ols.summary_frame()["obs_ci_lower"]
        iv_u = pred_ols.summary_frame()["obs_ci_upper"]

        f, ax = plt.subplots()
        ax.plot(df['consecutive_rewards'], df[key], "o", label="data")
        ax.plot(df['consecutive_rewards'], results.fittedvalues, "r--.", label="OLS")
        ax.plot(df['consecutive_rewards'], iv_u, "r--")
        ax.plot(df['consecutive_rewards'], iv_l, "r--")
        ax.legend(loc="best")
        plt.show()

    return results.params['a']


def collect_regression_coefficients(df: pd.DataFrame, sess_IDs: Iterable[Any]) -> np.ndarray:
    regression_coefs = []
    for sess in sess_IDs:
        ix = df['session_ID'] == sess
        _, state_df = vdp.count_consecutive_events(df[ix])
        coef = session_stats(state_df, key='trials_to_correct')
        regression_coefs.append(coef)
    return np.array(regression_coefs)


def compare_regression_coefficients(df: pd.DataFrame):
    sessions = np.unique(df['session_ID'])
    coefs = collect_regression_coefficients(df, sessions)

    markersize = 8
    f, ax = plt.subplots(figsize=(4, 4))
    plt.scatter(np.ones_like(HMM_coefs), HMM_coefs, c='w', label='Inference', s=markersize)
    plt.xticks([0, 1], ['RL', 'Inference'])
    # plt.ylim([np.amin(HMM_coefs)-.05, 1])
    plt.xlim([-.1, 1.1])

    # plt.legend()
    plt.title('Regression Coefficients by Model')
    plt.tight_layout()

    # plt.show()
    figure_format = 'png'
    f.savefig('../figures/{}.{}'.format('Vertechi2020_2E', figure_format), format=figure_format, dpi=300)
    print('Saved Vertechi 2020 Figure 2E')

