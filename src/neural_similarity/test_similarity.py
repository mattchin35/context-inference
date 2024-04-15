import src.neural_similarity.similarity_measures as sm
import src.neural_similarity.sample_data as sd
# import similarity_measures as sm
# import sample_data as sd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic


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
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(Y[:, 0], Y[:, 1])
    plt.title("Orthogonal samples")
    plt.show()


def parallel_samples():
    X, Y = sd.parallel_samples()
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(Y[:, 0], Y[:, 1])
    plt.title("Parallel samples")
    plt.show()


# def test_komorowski_2013():
#     f, ax = plt.subplots(3, 1, figsize=(8, 10))
#
#     plt.sca(ax[0])
#     X, Y = sd.orthogonal_samples(n_samples=10)
#     sim_xx, sim_yy, sim_xy = sm.komorowski_2013(X, Y)
#     sns.histplot(sim_xy, legend=True)
#     sns.histplot(sim_xx)
#     sns.histplot(sim_yy)
#     plt.title("Komorowski 2013 - cosine similarity between orthogonal samples")
#
#     plt.sca(ax[1])
#     X, Y = sd.parallel_samples()
#     sim_xx, sim_yy, sim_xy = sm.komorowski_2013(X, Y)
#     sns.histplot(sim_xy)
#     sns.histplot(sim_xx)
#     sns.histplot(sim_yy)
#     plt.title("Komorowski 2013 - cosine similarity between parallel samples")
#
#     plt.sca(ax[2])
#     X, Y = sd.uncorrelated_samples()
#     sim_xx, sim_yy, sim_xy = sm.komorowski_2013(X, Y)
#     sns.histplot(sim_xy)
#     sns.histplot(sim_xx)
#     sns.histplot(sim_yy)
#     plt.title("Komorowski 2013 - cosine similarity between uncorrelated samples")
#
#     plt.show()


# def test_plitt_2021():
#     f, ax = plt.subplots(3, 1, figsize=(8, 10))
#
#     plt.sca(ax[0])
#     X, Y = sd.orthogonal_samples(n_samples=10)
#     sim_xx, sim_yy, sim_xy = sm.plitt_2021(X, Y)
#     # sns.heatmap(sim_xy)
#     sns.histplot(sim_xy, legend=True)
#     sns.histplot(sim_xx)
#     sns.histplot(sim_yy)
#     plt.title("Plitt 2021 - cosine similarity between orthogonal samples")
#
#     plt.sca(ax[1])
#     X, Y = sd.parallel_samples()
#     sim_xx, sim_yy, sim_xy = sm.plitt_2021(X, Y)
#     # sns.heatmap(sim_xy)
#     sns.histplot(sim_xy)
#     sns.histplot(sim_xx)
#     sns.histplot(sim_yy)
#     plt.title("Plitt 2021 - cosine similarity between parallel samples")
#
#     plt.sca(ax[2])
#     X, Y = sd.uncorrelated_samples()
#     sim_xx, sim_yy, sim_xy = sm.plitt_2021(X, Y)
#     # sns.heatmap(sim_xy)
#     sns.histplot(sim_xy)
#     sns.histplot(sim_xx)
#     sns.histplot(sim_yy)
#     plt.title("Plitt 2021 - cosine similarity between uncorrelated samples")
#
#     plt.show()


# def test_kernel_similarity():
#     X, Y = sd.orthogonal_samples(n_samples=10)
#     orth_sim_xx, orth_sim_yy, orth_sim_xy = sm.kernel_alignment(X, Y)
#     # ic(sim_xx, sim_yy, sim_xy)
#
#     X, Y = sd.parallel_samples()
#     par_sim_xx, par_sim_yy, par_sim_xy = sm.kernel_alignment(X, Y)
#     # plt.title("Kernel similarity between orthogonal samples")
#
#     X, Y = sd.uncorrelated_samples()
#     uncor_sim_xx, uncor_sim_yy, uncor_sim_xy = sm.kernel_alignment(X, Y)
#
#     f, ax = plt.subplots()
#     plt.plot([orth_sim_xx, par_sim_xx, uncor_sim_xx], label="X vs X")
#     plt.plot([orth_sim_yy, par_sim_yy, uncor_sim_yy], label="Y vs Y")
#     plt.plot([orth_sim_xy, par_sim_xy, uncor_sim_xy], label="X vs Y")
#     plt.legend()
#     plt.xticks([0, 1, 2], ["Orthogonal", "Parallel", "Uncorrelated"])
#     plt.title("Kernel similarity")
#     plt.show()


if __name__ == "__main__":
    vector_cosine_similarity()
    matrix_cosine_similarity()
    orthogonal_samples()
    parallel_samples()

    # test_kernel_similarity()
    # test_matrix_cosine_similarity()
    # test_komorowski_2013()
    # plt.show()
    print("All tests passed!")