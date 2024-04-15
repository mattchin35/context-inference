import src.neural_similarity.similarity_measures as sm
import src.neural_similarity.abstraction_sample_data as asd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic
from typing import List, Tuple, Iterable, Union
import pandas as pd
import scipy as sp
import itertools
from more_itertools import powerset
from pathlib import Path

data_dir = Path('../../../data/processed/rnn_experiments')
model_dir = Path('../../../saved_models/RNN')
plot_path = Path('../../../reports/figures/model_behavior')

rng = np.random.default_rng(seed=12345)
# rng = np.random.default_rng()


def generate_sample_data():
    conditions = ["R1", "R2", "L1", "L2"]
    # data = asd.context_only_3d(cov=.1, jitter=.1)
    # data = asd.context_geometry(cov=.1, jitter=.1)
    # data = asd.fully_separable(cov=.1, jitter=.1)
    data = asd.random_3d(cov=.1)
    df_list = [pd.DataFrame(d) for d in data]
    [df.insert(0, "condition", [cond] * df.shape[0]) for df, cond in zip(df_list, conditions)]
    data = pd.concat(df_list, ignore_index=True)
    data_zscored = sp.stats.zscore(data.drop(columns="condition"), axis=0)
    data_zscored = pd.DataFrame(data_zscored, columns=data.columns[1:])
    data_zscored.insert(0, "condition", data["condition"])
    centers = get_cluster_centers(data_zscored)
    resampled = resample_clusters(data_zscored, n_samples=10000)

    # test shattering_dimensionality
    dichotomy_names = ["context", "stimulus"]
    dichotomies = ((('R1', 'R2'), ('L1', 'L2')),
                   (('R1', 'L1'), ('R2', 'L2')))
    sm.shattering_dimensionality(data=data_zscored, dichotomies=dichotomies)

    # test CCGP
    # context_split = [['R1', 'R2', 'R3'], ['L1', 'L2', 'L3']]
    # create dichotomies for context and stimulus
    context_split = [['R1', 'R2'], ['L1', 'L2']]
    stimulus_split = [['R1', 'L1'], ['R2', 'L2']]
    context_train, context_test = condition_dichotomies(context_split[0], context_split[1])
    stimulus_train, stimulus_test = condition_dichotomies(stimulus_split[0], stimulus_split[1])

    ic(context_train, context_test)

    contextA_train, contextB_train = split_conditions(context_train)
    contextA_test, contextB_test = split_conditions(context_test)
    groups = dict(trainA=contextA_train, trainB=contextB_train, testA=contextA_test, testB=contextB_test)
    # sm.CCGP(data=data_zscored, dichotomy_groups=groups)
    # sm.CCGP(data=centers, dichotomy_groups=groups)
    sm.CCGP(data=resampled, dichotomy_groups=groups)


def get_cluster_centers(data: pd.DataFrame):
    conditions = data["condition"].unique()
    centers = pd.DataFrame([data[data["condition"] == cond].mean(axis=0, numeric_only=True) for cond in conditions])
    centers.insert(0, "condition", conditions)
    return centers


def resample_clusters(data: pd.DataFrame, n_samples: int = 10000):
    conditions = data["condition"].unique()
    resampled = []
    cond_col = []
    for cond in conditions:
        cluster = data[data["condition"] == cond].drop(columns="condition")
        # resampled.append(sp.stats.multivariate_normal.rvs(mean=cluster.mean(axis=0), cov=cluster.cov(), size=n_samples))
        resampled.append(pd.DataFrame(rng.choice(cluster, size=n_samples, replace=True)))
        resampled[-1].insert(0, "condition", [cond] * n_samples)
    resampled = pd.concat(resampled, ignore_index=True)
    return resampled


def condition_dichotomies(cond_a: List[str], cond_b: List[str]):
    # collect the conditions in each group
    conditions = cond_a + cond_b

    # powerset of the conditions, excluding the empty and full set
    pset_a = list(powerset(cond_a))
    pset_b = list(powerset(cond_b))
    pset_a.pop(0)
    pset_b.pop(0)
    pset_a.pop(-1)
    pset_b.pop(-1)

    # group sets by the number of conditions
    cond_a_groups = [list(g) for k, g in itertools.groupby(pset_a, key=len)]
    cond_b_groups = [list(g) for k, g in itertools.groupby(pset_b, key=len)]
    conditions_train = []
    conditions_test = []
    for x, y in zip(cond_a_groups, cond_b_groups):
        train = np.array(list(itertools.product(x, y)))
        train = np.array([t.flatten() for t in train])
        test = [conditions.copy() for _ in train]
        for i, t in enumerate(train):
            for cond in t:
                test[i].remove(cond)

        for tr, te in zip(train, test):
            conditions_train.append(tr)
            conditions_test.append(np.array(te))
    conditions_train = np.array(conditions_train, dtype=object)
    conditions_test = np.array(conditions_test, dtype=object)
    return conditions_train, conditions_test


def split_conditions(conditions: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    groupA = []
    groupB = []
    for cond in conditions:
        split_size = int(cond.shape[0] / 2)
        groupA.append(cond[:split_size])
        groupB.append(cond[split_size:])
    return np.array(groupA), np.array(groupB)


def main():
    generate_sample_data()


if __name__ == "__main__":
    main()
