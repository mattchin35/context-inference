import similarity_measures as sm
import sample_data as sd
import abstraction_sample_data as asd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic
import itertools
from more_itertools import powerset
from typing import List, Tuple


def vector_cosine_similarity():
    X = np.array([1, 0])
    Y = np.array([0, 1])
    assert sm.vector_cosine_similarity(X, Y) == 0.0
    assert sm.vector_cosine_similarity(X, X) == 1.0


def matrix_cosine_similarity():
    X = np.array([[1, 0], [0, 1]])
    Y = np.array([[0, 1], [1, 0]])
    assert np.allclose(sm.matrix_cosine_similarity(X, Y), np.array([[0, 1], [1, 0]]))
    assert np.allclose(sm.matrix_cosine_similarity(X, X), np.array([[1, 0], [0, 1]]))


def orthogonal_samples():
    X, Y = sd.orthogonal_samples()
    f, ax = plt.subplots()
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(Y[:, 0], Y[:, 1])
    plt.title("Orthogonal samples")
    # plt.show()


def parallel_samples():
    X, Y = sd.parallel_samples()
    f, ax = plt.subplots()
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(Y[:, 0], Y[:, 1])
    plt.title("Parallel samples")
    # plt.show()


def context_2d():
    R1, R2, L1, L2 = asd.context_only_2d(cov=.1)
    f, ax = plt.subplots()
    plt.scatter(R1[:, 0], R1[:, 1], label="R1")
    plt.scatter(R2[:, 0], R2[:, 1], label="R2")
    plt.scatter(L1[:, 0], L1[:, 1], label="L1")
    plt.scatter(L2[:, 0], L2[:, 1], label="L2")
    plt.legend()
    plt.title("Context only 2D")


def context_3d():
    R1, R2, L1, L2 = asd.context_only_3d(cov=.1)
    f = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(R1[:, 0], R1[:, 1], R1[:, 2], label="R1")
    ax.scatter(R2[:, 0], R2[:, 1], R2[:, 2], label="R2")
    ax.scatter(L1[:, 0], L1[:, 1], L1[:, 2], label="L1")
    ax.scatter(L2[:, 0], L2[:, 1], L2[:, 2], label="L2")
    ax.legend()
    plt.title("Context only 3D")
    # plt.show()


def stimulus_3d():
    R1, R2, L1, L2 = asd.stimulus_only_3d(cov=.1)
    f = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(R1[:, 0], R1[:, 1], R1[:, 2], label="R1")
    ax.scatter(R2[:, 0], R2[:, 1], R2[:, 2], label="R2")
    ax.scatter(L1[:, 0], L1[:, 1], L1[:, 2], label="L1")
    ax.scatter(L2[:, 0], L2[:, 1], L2[:, 2], label="L2")
    ax.legend()
    plt.title("Stimulus only 3D")
    # plt.show()


def random_2d():
    R1, R2, L1, L2 = asd.random_2d(cov=.1)
    f, ax = plt.subplots()
    plt.scatter(R1[:, 0], R1[:, 1], label="R1")
    plt.scatter(R2[:, 0], R2[:, 1], label="R2")
    plt.scatter(L1[:, 0], L1[:, 1], label="L1")
    plt.scatter(L2[:, 0], L2[:, 1], label="L2")
    plt.legend()
    plt.title("Random 2D")


def random_3d():
    R1, R2, L1, L2 = asd.random_3d(cov=.1)
    f = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(R1[:, 0], R1[:, 1], R1[:, 2], label="R1")
    ax.scatter(R2[:, 0], R2[:, 1], R2[:, 2], label="R2")
    ax.scatter(L1[:, 0], L1[:, 1], L1[:, 2], label="L1")
    ax.scatter(L2[:, 0], L2[:, 1], L2[:, 2], label="L2")
    ax.legend()
    plt.title("Random 3D")


def context_geometry():
    R1, R2, L1, L2 = asd.context_geometry(cov=.1, jitter=5)
    f = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(R1[:, 0], R1[:, 1], R1[:, 2], label="R1")
    ax.scatter(R2[:, 0], R2[:, 1], R2[:, 2], label="R2")
    ax.scatter(L1[:, 0], L1[:, 1], L1[:, 2], label="L1")
    ax.scatter(L2[:, 0], L2[:, 1], L2[:, 2], label="L2")
    ax.legend()
    plt.title("Context geometry")


def stimulus_geometry():
    R1, R2, L1, L2 = asd.stimulus_geometry(cov=.1)
    f = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(R1[:, 0], R1[:, 1], R1[:, 2], label="R1")
    ax.scatter(R2[:, 0], R2[:, 1], R2[:, 2], label="R2")
    ax.scatter(L1[:, 0], L1[:, 1], L1[:, 2], label="L1")
    ax.scatter(L2[:, 0], L2[:, 1], L2[:, 2], label="L2")
    ax.legend()
    plt.title("Stimulus geometry")


def fully_separable():
    R1, R2, L1, L2 = asd.fully_separable(cov=.1)
    f = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(R1[:, 0], R1[:, 1], R1[:, 2], label="R1")
    ax.scatter(R2[:, 0], R2[:, 1], R2[:, 2], label="R2")
    ax.scatter(L1[:, 0], L1[:, 1], L1[:, 2], label="L1")
    ax.scatter(L2[:, 0], L2[:, 1], L2[:, 2], label="L2")
    ax.legend()
    plt.title("Fully separable")


def main():
    # vector_cosine_similarity()
    # matrix_cosine_similarity()
    # orthogonal_samples()
    # parallel_samples()

    # context_2d()
    # context_3d()
    # stimulus_3d()
    # random_2d()
    # random_3d()

    context_geometry()
    # stimulus_geometry()
    # fully_separable()

    # prepare_data()
    plt.show()


if __name__ == "__main__":
    main()

