import numpy as np
import scipy as sp
# from scipy.stats import norm, multivariate_normal
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Protocol, Union


"""
Create sample data for testing of similiarity measures.

Make an R+/R-/L+/L- set with 50 samples each.
- one optimized for context, i.e. R+=R-, L+=L-
- one separating context and stimulus in a plane
- one separating them
"""


def orthogonal_samples(n_samples: int = 10):
    rng = np.random.default_rng()
    X = np.zeros((n_samples, 2))
    # X[:, 0] = rng.random(size=n_samples) * 100
    X[:, 0] = sp.stats.norm.rvs(loc=50, scale=1, size=n_samples)
    X[:, 1] = sp.stats.norm.rvs(loc=0, scale=1, size=n_samples)

    Y = np.zeros((n_samples, 2))
    Y[:, 0] = sp.stats.norm.rvs(loc=0, scale=1, size=n_samples)
    # Y[:, 1] = rng.random(size=n_samples) * 100
    Y[:, 1] = sp.stats.norm.rvs(loc=50, scale=1, size=n_samples)
    return X, Y


def parallel_samples(n_samples: int = 10):
    """Align samples along 45 degrees"""
    rng = np.random.default_rng()
    X = np.zeros((n_samples, 2))
    Y = np.zeros((n_samples, 2))
    for i in range(n_samples):
        X[i] = sp.stats.norm.rvs(loc=rng.integers(100), scale=1, size=2)
        Y[i] = sp.stats.norm.rvs(loc=rng.integers(100), scale=1, size=2)
    return X, Y


def correlated_samples(n_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    X = np.zeros((n_samples, 2))
    for i in range(n_samples):
        X[i] = sp.stats.norm.rvs(loc=rng.integers(100), scale=1, size=2)
    return X[:, 0], X[:, 1]


def uncorrelated_samples(n_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    X = rng.random(size=(n_samples, 2)) * 100
    Y = rng.random(size=(n_samples, 2)) * 100
    return X, Y


