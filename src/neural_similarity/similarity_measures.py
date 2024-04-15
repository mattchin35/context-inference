import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Protocol, Union, List, Iterable
from icecream import ic
from sklearn import decomposition, svm
from sklearn.model_selection import train_test_split
# from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler
# import src.neural_similarity.abstraction_sample_data as asd


def vector_cosine_similarity(X: np.ndarray, Y: np.ndarray):
    """X and Y are 1D vectors of the same length."""
    return np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y))


def matrix_cosine_similarity(X: np.ndarray, Y: np.ndarray):
    """
    Calculate cosine similarity between each pair of vectors in X and Y.
    X and Y are matrices with shape (n_samples, n_features)
    """
    Xnorm = np.linalg.norm(X, axis=1)
    Ynorm = np.linalg.norm(Y, axis=1)
    similarity = np.dot(X, Y.T) / np.outer(Xnorm, Ynorm)

    # equivalent formula, used in plitt_2021
    # Xnorm = (X.T / np.linalg.norm(X, axis=1)).T
    # Ynorm = (Y.T / np.linalg.norm(Y, axis=1)).T
    # similarity = np.dot(Xnorm, Ynorm.T)
    return similarity


def similarity_fraction(X: np.ndarray, standard1: np.ndarray, standard2: np.ndarray):
    """
    An index of similarity between X and two standards.
    From Plitt and Giocomo, Nature Neuroscience 2021
    """
    Xnorm = np.linalg.norm(X, axis=1)
    sim1 = np.dot(X, standard1) / (Xnorm * np.linalg.norm(standard1))
    sim2 = np.dot(X, standard2) / (Xnorm * np.linalg.norm(standard2))
    return sim1 / (sim1 + sim2)


def komorowski_2013(X: np.ndarray, Y: np.ndarray):
    """
    Methods based on Komorowski et al., J Neuroscience 2013
    Each matrix is trials x neurons
    Normalize each column (neuron) by z-score
    Calculate cosine similarity between each pair of trials
    Separate cosine similarities by group to perform statistics - condition 1 vs 2, stimulus vs no stimulus, correct vs incorrect, etc.
    """

    # Normalize each column (neuron) by z-score
    N1, N2 = X.shape[0], Y.shape[0]
    zscored = sp.stats.zscore(np.concatenate((X, Y), axis=0), axis=0)
    Xz, Yz = zscored[:N1], zscored[N1:]

    # Calculate cosine similarity between each pair of trials
    cos_sim_xx = matrix_cosine_similarity(Xz, Xz)
    cos_sim_yy = matrix_cosine_similarity(Yz, Yz)
    cos_sim_xy = matrix_cosine_similarity(Xz, Yz)

    # for xx and yy, the diagonal should be 1 and the matrix should be symmetric
    # take the upper triangle of the matrix, excluding the diagonal
    mask = np.triu(np.ones(cos_sim_xx.shape, dtype=bool), k=1)
    cos_sim_xx = cos_sim_xx[mask]
    cos_sim_yy = cos_sim_yy[mask]
    cos_sim_xy = cos_sim_xy.flatten()

    return cos_sim_xx, cos_sim_yy, cos_sim_xy


def plitt_2021(X: np.ndarray, Y: np.ndarray):
    """
    Methods based on Plitt and Giocomo, Nature Neuroscience 2021
    Each matrix is trials x neurons
    Normalize each column (neuron) by L2 norm
    Calculate cosine similarity between each pair of trials
    Separate cosine similarities by group to perform statistics - condition 1 vs 2, stimulus vs no stimulus, correct vs incorrect, etc.

    Need to extend this to either have no groups or have more than 2 groups
    """

    # Normalize each column (neuron) by L2 norm
    N1, N2 = X.shape[0], Y.shape[0]
    tmp = np.concatenate((X, Y), axis=0)
    normalized = tmp / np.linalg.norm(tmp, axis=0)
    Xz, Yz = normalized[:N1], normalized[N1:]

    # Calculate cosine similarity between each pair of trials
    cos_sim_xx = matrix_cosine_similarity(Xz, Xz)
    cos_sim_yy = matrix_cosine_similarity(Yz, Yz)
    cos_sim_xy = matrix_cosine_similarity(Xz, Yz)
    cos_sim_all = matrix_cosine_similarity(normalized, normalized)

    # for xx and yy, the diagonal should be 1 and the matrix should be symmetric
    # take the upper triangle of the matrix, excluding the diagonal
    mask = np.triu(np.ones(cos_sim_xx.shape, dtype=bool), k=1)
    cos_sim_xx = cos_sim_xx[mask]
    cos_sim_yy = cos_sim_yy[mask]
    cos_sim_xy = cos_sim_xy.flatten()

    # PCA
    norm_std = StandardScaler().fit_transform(cos_sim_all)
    pca = decomposition.PCA(n_components=2)
    pca.fit(norm_std)

    return cos_sim_xx, cos_sim_yy, cos_sim_xy


def parallelism_score(X: np.ndarray, Y: np.ndarray):
    pass


def shattering_dimensionality(data: pd.DataFrame, dichotomies: Iterable) -> Tuple[float, List]:
    """
    Shattering dimensionality assesses the ability to
    decode all possible dichotomies of the data. Within each dichotomy,
    all experimental conditions are used with a typical train-test split.

    Will need to adjust this in the future for different numbers of units for each dichotomy.
    """
    scores = []
    for A, B in dichotomies:
        data_A = data[data["condition"].isin(A)]
        data_B = data[data["condition"].isin(B)]
        # data_A = data_zscored[data_zscored["condition"].isin(A)]
        # data_B = data_zscored[data_zscored["condition"].isin(B)]
        data_A = data_A.drop(columns="condition")
        data_B = data_B.drop(columns="condition")
        labels = data_A.shape[0] * [0] + data_B.shape[0] * [1]
        training_data = pd.concat([data_A, data_B], ignore_index=True)

        # can change this to do multiple splits and average the scores
        X_train, X_test, y_train, y_test = train_test_split(training_data.to_numpy(), labels, test_size=0.2)#, random_state=42)
        clf = svm.SVC(kernel="linear")
        clf.fit(X_train, y_train)

        # pred = clf.predict(X_test)
        score = clf.score(X_test, y_test)
        scores.append(score)

    SD = np.mean(scores)
    ic(scores)
    print('Shattering dimensionality: {}'.format(SD))
    return SD, scores


def CCGP(data: pd.DataFrame, dichotomy_groups: dict, train_type: str='center') -> Tuple[float, List]:
    """
    Cross condition generalization performance.
    data should already be z-scored when given to this function
    Train and test decoding performance with different condition sets from each side of the dichotomy
    Use SVM (maximum margin linear classifier)
    Compare to linear classifier for each condition and overall shattering dimensionality
    Only train on correct trials!!
    https://bit.ly/4a4QZNw for SVM explanation
    """
    assert train_type in ['center', 'resample']

    scores = []
    nsplits = dichotomy_groups["trainA"].shape[0]
    for i in range(nsplits):
        trainA = data[data["condition"].isin(dichotomy_groups["trainA"][i])]
        trainB = data[data["condition"].isin(dichotomy_groups["trainB"][i])]
        trainA = trainA.drop(columns="condition")
        trainB = trainB.drop(columns="condition")

        train_labels = trainA.shape[0] * [0] + trainB.shape[0] * [1]
        training_data = pd.concat([trainA, trainB], ignore_index=True)

        testA = data[data["condition"].isin(dichotomy_groups["testA"][i])]
        testB = data[data["condition"].isin(dichotomy_groups["testB"][i])]
        testA = testA.drop(columns="condition")
        testB = testB.drop(columns="condition")
        test_labels = testA.shape[0] * [0] + testB.shape[0] * [1]
        test_data = pd.concat([testA, testB], ignore_index=True)

        clf = svm.SVC(kernel="linear")
        clf.fit(training_data, train_labels)
        
        # pred = clf.predict(X_test)
        score = clf.score(test_data, test_labels)
        scores.append(score)

    cross_condition_generalization = np.mean(scores)
    ic(scores)
    print('CCGP: {}'.format(cross_condition_generalization))
    return cross_condition_generalization, scores


def kernel_inner_product(k1: np.ndarray, k2: np.ndarray):
    # the following are equivalent ways to calculate the kernel inner product
    # pick whichever is faster or more readable
    prod = np.trace(np.dot(k1, k2))
    # prod = np.sum(k1 * k2)
    # prod = np.dot(k1.flatten(), k2.flatten())
    return prod


def kernel_alignment(X: np.ndarray, Y: np.ndarray):
    # used in Alleman ICLR 2024, from Cristianini NIPS 2001
    # can be used to compare sessions, but only one set of values per session
    Kx = np.dot(X.T, X)
    Ky = np.dot(Y.T, Y)
    alignment_xx = kernel_inner_product(Kx, Kx) / np.sqrt(kernel_inner_product(Kx, Kx) * kernel_inner_product(Kx, Kx))
    alignment_yy = kernel_inner_product(Ky, Ky) / np.sqrt(kernel_inner_product(Ky, Ky) * kernel_inner_product(Ky, Ky))
    alignment_xy = kernel_inner_product(Kx, Ky) / np.sqrt(kernel_inner_product(Kx, Kx) * kernel_inner_product(Ky, Ky))
    return alignment_xx, alignment_yy, alignment_xy


def main():
    shattering_dimensionality()
    # ic(vector_cosine_similarity(np.array([1, -1]), np.array([-1, 1])))
    #
    # X, Y = sd.orthogonal_samples(n_samples=10)
    # sim_xx, sim_yy, sim_xy = plitt_2021(X, Y)

if __name__ == "__main__":
    main()

