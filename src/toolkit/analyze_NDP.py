import jax.numpy as jnp
import numpy as onp
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import time
def analyze_graph(weights, mask):

    weights = jnp.where(mask[:, None], weights, 0 )
    weights = jnp.where(mask[None,:], weights, 0 )

    thres = 0.0001
    weights =jnp.where(jnp.abs(weights)< thres, 0, weights)

    # number of nodes
    num_nodes = int(jnp.count_nonzero(
        jnp.sum(weights, axis=0) + jnp.sum(weights, axis=1)))

    # number of edges
    num_edges = int(jnp.count_nonzero(weights))

    # adjacency matrix
    adj_matrix = jnp.where(weights, 1, 0)

    # density
    density = float(num_edges / jnp.size(weights))

    # diversity
    diversity = float(jnp.var(weights))

    # clustering
    graph = nx.from_numpy_array(onp.array(weights))
    clustering = float(nx.average_clustering(graph))

    # small-worldness
    """
    try:
        start =time.time()
        if nx.is_connected(graph):
            sigma = nx.sigma(graph)
            print("sw sigma utils took " + str(time.time()-start))
            start =time.time()

            omega = nx.omega(graph)
            print("sw omega utils took " + str(time.time()-start))
        else:
            sigma = 0
            omega = 1
    

    except nx.exception.NetworkXError:
        sigma = 0
        omega = 1
    """
    sigma = 0
    omega=1


    # degree histogram'
    degrees = [graph.degree(n) for n in graph.nodes()]

    high_degree = int(len(graph.nodes)*0.4)
    high_degree_nodes = onp.sum(degrees[high_degree:])

    graph_features = {"num_nodes": num_nodes,
                      "num_edges": num_edges,
                      "adj_matrix": adj_matrix,
                      "density": density,
                      "weights": weights,
                      "clustering": clustering,
                      "diversity": diversity,
                      "sw_sigma": sigma,
                      "sw_omega": omega,
                      "high_degree_nodes": high_degree_nodes}
    return graph_features

def analyze_hidden(hidden):
    hidden = onp.array(hidden)

    # compute diversity
    pca = PCA()
    pca.fit(hidden)
    k = onp.argmax(onp.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1

    if k < 2:
        k = 2
    transformed_data = pca.transform(hidden)[:, :k]

    mean_vector = onp.mean(transformed_data, axis=0)
    deviation_matrix = transformed_data - mean_vector
    covariance_matrix = onp.cov(deviation_matrix, rowvar=False)  # rowvar=False assumes each column is a variable
    diversity = float(onp.trace(covariance_matrix))

    def knn_sparsity(x, k=10):
        dists = jnp.sqrt(jnp.sum(x[:, None, :] - x[None, :, :], axis=-1) ** 2)
        res = 0.
        for i in range(x.shape[0]):
            idxs = jnp.argsort(dists[i])
            knn = idxs[1:k + 1]
            res += jnp.mean(dists[i, knn])
        return res / x.shape[0]

    diversity_simple = knn_sparsity(hidden)

    tsne = TSNE(n_components=2, random_state=42)
    embedded_data = tsne.fit_transform(hidden)

    kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust the number of clusters as needed
    cluster_labels = kmeans.fit_predict(hidden)

    hidden_features = {"diversity": diversity,
                       "diversity_simple": diversity_simple,
                       "hidden": hidden,
                       "tsne": [embedded_data, cluster_labels]}


    # Step 3: Create a scatter plot with different colors for each cluster

    return hidden_features




