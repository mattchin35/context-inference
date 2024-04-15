import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Protocol, Union

rng = np.random.default_rng(seed=12345)
# rng = np.random.default_rng()


def center(jitter: float = 0.1):
    return rng.random() * 2 * jitter - jitter


def context_only_2d(n_samples: int = 50, jitter: float = 0.1, cov: float = 1):
    """Create a set of samples optimized for context, i.e. R+=R-, L+=L-"""
    R1 = sp.stats.multivariate_normal.rvs(mean=[1, center(jitter)], cov=cov, size=n_samples)
    R2 = sp.stats.multivariate_normal.rvs(mean=[1, center(jitter)], cov=cov, size=n_samples)
    L1 = sp.stats.multivariate_normal.rvs(mean=[-1, center(jitter)], cov=cov, size=n_samples)
    L2 = sp.stats.multivariate_normal.rvs(mean=[-1, center(jitter)], cov=cov, size=n_samples)
    return R1, R2, L1, L2


def context_only_3d(n_samples: int = 50, jitter: float = 0.1, cov: float = 1):
    """Create a set of samples optimized for context, i.e. R+=R-, L+=L-"""
    R1 = sp.stats.multivariate_normal.rvs(mean=[1, center(jitter), center(jitter)], cov=cov, size=n_samples)
    R2 = sp.stats.multivariate_normal.rvs(mean=[1, center(jitter), center(jitter)], cov=cov, size=n_samples)
    L1 = sp.stats.multivariate_normal.rvs(mean=[-1, center(jitter), center(jitter)], cov=cov, size=n_samples)
    L2 = sp.stats.multivariate_normal.rvs(mean=[-1, center(jitter), center(jitter)], cov=cov, size=n_samples)
    return R1, R2, L1, L2


def stimulus_only_3d(n_samples: int = 50, jitter: float = 0.1, cov: float = 1):
    R1 = sp.stats.multivariate_normal.rvs(mean=[center(jitter), 1, center(jitter)], cov=cov, size=n_samples)
    R2 = sp.stats.multivariate_normal.rvs(mean=[center(jitter), -1, center(jitter)], cov=cov, size=n_samples)
    L1 = sp.stats.multivariate_normal.rvs(mean=[center(jitter), 1, center(jitter)], cov=cov, size=n_samples)
    L2 = sp.stats.multivariate_normal.rvs(mean=[center(jitter), -1, center(jitter)], cov=cov, size=n_samples)
    return R1, R2, L1, L2


def random_2d(n_samples: int = 50, cov: float = 1):
    random_center = lambda: rng.random() * 4 - 2  # range between [-2, 2]
    R1 = sp.stats.multivariate_normal.rvs(mean=[random_center(), random_center()], cov=cov, size=n_samples)
    R2 = sp.stats.multivariate_normal.rvs(mean=[random_center(), random_center()], cov=cov, size=n_samples)
    L1 = sp.stats.multivariate_normal.rvs(mean=[random_center(), random_center()], cov=cov, size=n_samples)
    L2 = sp.stats.multivariate_normal.rvs(mean=[random_center(), random_center()], cov=cov, size=n_samples)
    return R1, R2, L1, L2


def random_3d(n_samples: int = 50, cov: float = 1):
    random_center = lambda: rng.random() * 4 - 2  # range between [-2, 2]
    R1 = sp.stats.multivariate_normal.rvs(mean=[random_center(), random_center(), random_center()], cov=cov, size=n_samples)
    R2 = sp.stats.multivariate_normal.rvs(mean=[random_center(), random_center(), random_center()], cov=cov, size=n_samples)
    L1 = sp.stats.multivariate_normal.rvs(mean=[random_center(), random_center(), random_center()], cov=cov, size=n_samples)
    L2 = sp.stats.multivariate_normal.rvs(mean=[random_center(), random_center(), random_center()], cov=cov, size=n_samples)
    return R1, R2, L1, L2


def context_geometry(n_samples: int = 50, jitter: float = 0.1, cov: float = 1):
    """Samples where context (R, L) is separable but stimulus (1,2) is not"""
    R1 = sp.stats.multivariate_normal.rvs(mean=[-1, 1, center(jitter)], cov=cov, size=n_samples)
    R2 = sp.stats.multivariate_normal.rvs(mean=[1, 1, center(jitter)], cov=cov, size=n_samples)
    L1 = sp.stats.multivariate_normal.rvs(mean=[1, -1, center(jitter)], cov=cov, size=n_samples)
    L2 = sp.stats.multivariate_normal.rvs(mean=[-1, -1, center(jitter)], cov=cov, size=n_samples)
    return R1, R2, L1, L2


def stimulus_geometry(n_samples: int = 50, jitter: float = 0.1, cov: float = 1):
    """Samples where stimulus (1,2) is separable but context (R,L) is not"""
    R1 = sp.stats.multivariate_normal.rvs(mean=[-1, 1, center(jitter)], cov=cov, size=n_samples)
    R2 = sp.stats.multivariate_normal.rvs(mean=[1, -1, center(jitter)], cov=cov, size=n_samples)
    L1 = sp.stats.multivariate_normal.rvs(mean=[1, 1, center(jitter)], cov=cov, size=n_samples)
    L2 = sp.stats.multivariate_normal.rvs(mean=[-1, -1, center(jitter)], cov=cov, size=n_samples)
    return R1, R2, L1, L2


def fully_separable(n_samples: int = 50, jitter: float = 0.1, cov: float = 1):
    """Samples where both context (R,L) and stimulus (1,2) are separable"""
    R1 = sp.stats.multivariate_normal.rvs(mean=[-1, 1, center(jitter)], cov=cov, size=n_samples)
    R2 = sp.stats.multivariate_normal.rvs(mean=[1, 1, center(jitter)], cov=cov, size=n_samples)
    L1 = sp.stats.multivariate_normal.rvs(mean=[-1, -1, center(jitter)], cov=cov, size=n_samples)
    L2 = sp.stats.multivariate_normal.rvs(mean=[1, -1, center(jitter)], cov=cov, size=n_samples)
    return R1, R2, L1, L2


def main():
    R1, R2, L1, L2 = context_only_2d()


if __name__ == "__main__":
    main()