# ===================================================
#  UMAP Fit and Transform Operations Test cases
#  (not really fitting anywhere else)
# ===================================================

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.preprocessing import normalize
from numpy.testing import assert_array_equal
from umap import UMAP
from umap.spectral import component_layout
import numpy as np
import scipy.sparse
import pytest
import warnings
from umap.distances import pairwise_special_metric
from umap.utils import disconnected_vertices
from scipy.sparse import csr_matrix

# Transform isn't stable under batching; hard to opt out of this.
# @SkipTest
# def test_scikit_learn_compatibility():
#     check_estimator(UMAP)


# This test is currently to expensive to run when turning
# off numba JITting to detect coverage.
# @SkipTest
# def test_umap_regression_supervision(): # pragma: no cover
#     boston = load_boston()
#     data = boston.data
#     embedding = UMAP(n_neighbors=10,
#                      min_dist=0.01,
#                      target_metric='euclidean',
#                      random_state=42).fit_transform(data, boston.target)
#


# Umap Clusterability
def test_blobs_cluster():
    data, labels = make_blobs(n_samples=500, n_features=10, centers=5)
    embedding = UMAP(n_epochs=100).fit_transform(data)
    assert adjusted_rand_score(labels, KMeans(5).fit_predict(embedding)) == 1.0


# Multi-components Layout
def test_multi_component_layout():
    data, labels = make_blobs(
        100, 2, centers=5, cluster_std=0.5, center_box=[-20, 20], random_state=42
    )

    true_centroids = np.empty((labels.max() + 1, data.shape[1]), dtype=np.float64)

    for label in range(labels.max() + 1):
        true_centroids[label] = data[labels == label].mean(axis=0)

    true_centroids = normalize(true_centroids, norm="l2")

    embedding = UMAP(n_neighbors=4, n_epochs=100).fit_transform(data)
    embed_centroids = np.empty((labels.max() + 1, data.shape[1]), dtype=np.float64)
    embed_labels = KMeans(n_clusters=5).fit_predict(embedding)

    for label in range(embed_labels.max() + 1):
        embed_centroids[label] = data[embed_labels == label].mean(axis=0)

    embed_centroids = normalize(embed_centroids, norm="l2")

    error = np.sum((true_centroids - embed_centroids) ** 2)

    assert error < 15.0, "Multi component embedding to far astray"


@pytest.mark.parametrize("num_isolates", [1])
@pytest.mark.parametrize("sparse", [True, False])
def test_disconnected_data_precomputed(num_isolates, sparse):
    disconnected_data = np.random.choice(
        a=[False, True], size=(10, 20), p=[0.66, 1 - 0.66]
    )
    # Add some disconnected data for the corner case test
    disconnected_data = np.vstack(
        [disconnected_data, np.zeros((num_isolates, 20), dtype="bool")]
    )
    new_columns = np.zeros((num_isolates + 10, num_isolates), dtype="bool")
    for i in range(num_isolates):
        new_columns[10 + i, i] = True
    disconnected_data = np.hstack([disconnected_data, new_columns])
    dmat = pairwise_special_metric(disconnected_data)
    if sparse:
        dmat = csr_matrix(dmat)
    model = UMAP(n_neighbors=3, metric="precomputed", disconnection_distance=1).fit(
        dmat
    )

    # Check that the first isolate has no edges in our umap.graph_
    isolated_vertices = disconnected_vertices(model)
    assert isolated_vertices[10] == True
    number_of_nan = np.sum(np.isnan(model.embedding_[isolated_vertices]))
    assert number_of_nan >= num_isolates * model.n_components


# -----------------
# UMAP Graph output
# -----------------
def test_umap_graph_layout():
    data, labels = make_blobs(n_samples=500, n_features=10, centers=5)
    model = UMAP(n_epochs=100, transform_mode="graph")
    graph = model.fit_transform(data)
    assert scipy.sparse.issparse(graph)
    nc, cl = scipy.sparse.csgraph.connected_components(graph)
    assert nc == 5

    new_graph = model.transform(data[:10] + np.random.normal(0.0, 0.1, size=(10, 10)))
    assert scipy.sparse.issparse(graph)
    assert new_graph.shape[0] == 10
