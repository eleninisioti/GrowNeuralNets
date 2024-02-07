import matplotlib.pyplot as plt
import os
import jax.numpy as jnp
import numpy as onp
import seaborn
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


# ----- general figure configuration -----
width, height = set_size(241.14749)
fig_size = (width, height )
params = {'legend.fontsize': 11,
          "figure.autolayout": True,
          'font.size': 11,
          "figure.figsize": fig_size}
plt.rcParams.update(params)


def pastel(color):
    return tuple((x + 1.0) / 2.0 for x in color)


def viz_graph_growth(save_dir, growth_features):
    save_dir = save_dir + "/graph_features"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    growth_duration = len(growth_features)
    temp = [el["num_nodes"] for el in growth_features]
    plt.plot(range(growth_duration), [el["num_nodes"] for el in growth_features], label="#nodes")
    plt.xlabel("steps")
    plt.ylabel("#Nodes")

    plt.tight_layout()
    plt.savefig(save_dir + "/num_nodes.png")
    plt.clf()

    plt.plot(range(growth_duration), [el["num_edges"] for el in growth_features], label="#edges")
    plt.xlabel("Steps")
    plt.ylabel("#Edges")
    plt.tight_layout()
    plt.savefig(save_dir + "/num_edges.png")
    plt.clf()

    plt.plot(range(growth_duration), [el["density"] for el in growth_features], label="density")
    plt.xlabel("Steps")
    plt.ylabel("Density")

    plt.tight_layout()
    plt.savefig(save_dir + "/density.png")
    plt.clf()

    plt.plot(range(growth_duration), [el["clustering"] for el in growth_features], label="clustering")
    plt.xlabel("Steps")
    plt.ylabel("Clustering")
    plt.tight_layout()
    plt.savefig(save_dir + "/clustering.png")
    plt.clf()

    plt.plot(range(growth_duration), [el["sw_sigma"] for el in growth_features], label="sw_sigma")
    plt.xlabel("Steps")
    plt.ylabel("Sw-sigma")
    plt.tight_layout()
    plt.savefig(save_dir + "/sw_sigma.png")
    plt.clf()

    plt.plot(range(growth_duration), [el["sw_omega"] for el in growth_features], label="sw_omega")
    plt.xlabel("Steps")
    plt.ylabel("Sw-omega")
    plt.tight_layout()
    plt.savefig(save_dir + "/sw_omega.png")
    plt.clf()

    plt.plot(range(growth_duration), [el["high_degree_nodes"] for el in growth_features], label="high_degree_nodes")
    plt.xlabel("Steps")
    plt.ylabel("High degree nodes")
    plt.tight_layout()
    plt.savefig(save_dir + "/hd_nodes.png")
    plt.clf()

    diff_adj = []
    diff_weights = []
    for step in range(1, growth_duration):
        diff_adj.append(jnp.sum(growth_features[step]["adj_matrix"] != growth_features[step - 1]["adj_matrix"]))
        diff_weights.append(jnp.sum(jnp.abs(growth_features[step]["weights"] - growth_features[step - 1]["weights"])))

    plt.plot(range(1, growth_duration), diff_adj)
    plt.xlabel("Steps")
    plt.ylabel("Adjacency update")

    plt.tight_layout()
    plt.savefig(save_dir + "/diff_adj.png")
    plt.clf()

    plt.plot(range(1, growth_duration), diff_weights)
    plt.xlabel("Steps")
    plt.ylabel("Weights update")

    plt.tight_layout()
    plt.savefig(save_dir + "/diff_weights.png")
    plt.clf()


def viz_hidden_growth(save_dir, hidden_features):
    save_dir = save_dir + "/hidden_features"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    growth_duration = len(hidden_features)

    plt.plot(range(growth_duration), [el["diversity"] for el in hidden_features], label="diversity")
    plt.xlabel("Steps")
    plt.ylabel("Diversity")

    plt.tight_layout()
    plt.savefig(save_dir + "/diversity.png")
    plt.clf()

    for step, total_data in enumerate(hidden_features):
        step_data = total_data["tsne"][0]

        cluster_labels = total_data["tsne"][1]
        for i in range(max(cluster_labels)):
            plt.scatter(step_data[cluster_labels == i, 0], step_data[cluster_labels == i, 1], label=f'Cluster {i + 1}')

        ##            cmap='RdBu',s=50)  # Use a suitable colormap
        plt.savefig(save_dir + "/tsne_" + str(step) + ".png")
        plt.clf()

    diff_hidden = []
    for step in range(1, growth_duration):
        diff_hidden.append(jnp.sum(hidden_features[step]["hidden"] != hidden_features[step - 1]["hidden"]))

    plt.plot(range(1, growth_duration), diff_hidden)
    plt.xlabel("Steps")
    plt.ylabel("Hidden update")
    plt.tight_layout()
    plt.savefig(save_dir + "/diff_hidden.png")
    plt.clf()


def viz_hidden(project_dir, gen, growth_step, hidden_states):
    saving_directory = project_dir + "/gen_" + str(gen)

    if not os.path.exists(saving_directory + "/heatmaps_hidden"):
        os.makedirs(saving_directory + "/heatmaps_hidden")

    heatmap_filename = saving_directory + "/heatmaps_hidden/" + str(growth_step) + ".png"
    plt.clf()
    tmp = onp.array(hidden_states)
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size here
    heatmap = seaborn.heatmap(hidden_states, cmap='RdBu', fmt='.1e', cbar=True, annot_kws={"size": 6},
                              cbar_kws={"shrink": 0.7})
    y_ticks = range(0, len(hidden_states), 5)  # Set ticks every 5
    heatmap.set_yticks(y_ticks)
    heatmap.set_yticklabels(y_ticks)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.set_offset_position('left')
    plt.xlabel("Feature")
    plt.ylabel("Neurons")
    cbar.update_ticks()
    plt.tight_layout()
    plt.savefig(heatmap_filename, format='png')
    plt.clf()


def viz_weights(project_dir, gen, growth_step, weights):
    """ Plots heatmap of the weights of a neural network
    """
    saving_directory = project_dir + "/gen_" + str(gen)
    if not os.path.exists(saving_directory + "/heatmaps"):
        os.makedirs(saving_directory + "/heatmaps")

    # save heatmap of weights
    heatmap_filename = saving_directory + "/heatmaps/" + str(growth_step) + ".png"
    fig, ax = plt.subplots(figsize=fig_size)  # Adjust figure size here
    temp = onp.array(weights)
    heatmap = seaborn.heatmap(weights, cmap='RdBu', fmt='.1e', cbar=True,
                              annot_kws={"size": 6},
                              cbar_kws={"shrink": 0.7})
    xtick_frequency = 50  # Set the desired frequency
    ax.set_xticks(ax.get_xticks()[::xtick_frequency])
    ax.set_xticklabels(ax.get_xticklabels()[::xtick_frequency])
    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.set_offset_position('left')
    cbar.update_ticks()
    plt.tight_layout()
    plt.savefig(heatmap_filename, format='png', dpi=300)
    plt.clf()
    plt.close()


def scale_edge_widths(graph, weights):
    max_w = jnp.max(weights)
    max_edge_width = 1
    scaled = weights / (max_w+1)
    edge_widths = {}
    for edge in graph.edges:
        edge_widths[edge] = float(max_edge_width * scaled[edge[0], edge[1]])
    return edge_widths


def viz_network(project_dir, gen, growth_step, weights, hidden, mask, n_input_nodes, n_output_nodes, n_hidden_nodes):
    saving_directory = project_dir + "/gen_" + str(gen)
    if not os.path.exists(saving_directory + "/networks"):
        os.makedirs(saving_directory + "/networks")

    network_filename = saving_directory + "/networks/" + str(growth_step) + ".png"
    plt.clf()

    g = nx.Graph()
    g.add_nodes_from(range(n_input_nodes), color=pastel((1, 0, 0)))
    g.add_nodes_from(range(n_input_nodes, n_input_nodes + n_output_nodes), color=pastel((0, 0, 1)))
    g.add_nodes_from(range(n_input_nodes + n_output_nodes, n_input_nodes + n_hidden_nodes + n_output_nodes),
                     color=pastel((0, 1, 0)))

    total_nodes = n_input_nodes + n_hidden_nodes + n_output_nodes

    # remove edges whose weights are too low
    weight_low = 0.00001
    tmp = onp.array(weights)

    weights = jnp.where(jnp.abs(weights) < weight_low, 0, weights)

    tmp = onp.array(weights)

    edges = []
    nodes_to_remove = []
    for inode in range(total_nodes):

        n_connections = jnp.sum(weights[inode, :]) + jnp.sum(weights[:, inode])
        if n_connections == 0:
            nodes_to_remove.append(inode)
        else:
            for onode in range(total_nodes):
                if jnp.any(weights[inode, onode] != 0.0):
                    edges.append((inode, onode))

    mask = onp.array(mask).tolist()
    nodes_to_remove = [idx for idx, el in enumerate(mask) if el == 0]

    g.add_edges_from(edges, color="black")
    g.remove_nodes_from(nodes_to_remove)

    print("nodes to remove ", str(len(nodes_to_remove)))

    node_colors = [g.nodes[n]['color'] for n in g.nodes]
    edge_colors = [g.edges[e]['color'] for e in g.edges]

    node_size = 500

    edge_widths = scale_edge_widths(g, weights)
    edge_widths = [1 for e in edge_widths]
    #pca = PCA(n_components=2)
    #pos =pca.fit_transform(hidden)

    nx.draw(g, pos=nx.kamada_kawai_layout(g), node_color=node_colors, node_size=node_size, width=edge_widths,
            edge_color=edge_colors,
            with_labels=False)

    # Save the graph to a file (e.g., PNG, PDF, SVG, etc.)
    plt.savefig(network_filename, format='png')
    plt.clf()
    plt.close()
    num_nodes = len(g.nodes)
    num_edges = len(g.edges)

    return num_nodes, num_edges


def viz_network_colored(project_dir, gen, growth_step, weights, hidden, mask, n_input_nodes, n_output_nodes, n_hidden_nodes):
    saving_directory = project_dir + "/gen_" + str(gen)
    if not os.path.exists(saving_directory + "/networks"):
        os.makedirs(saving_directory + "/networks")

    network_filename = saving_directory + "/networks/" + str(growth_step) + "_colored.png"
    plt.clf()

    g = nx.Graph()
    g.add_nodes_from(range(n_input_nodes), color=pastel((1, 0, 0)))
    g.add_nodes_from(range(n_input_nodes, n_input_nodes + n_output_nodes), color=pastel((0, 0, 1)))
    g.add_nodes_from(range(n_input_nodes + n_output_nodes, n_input_nodes + n_hidden_nodes + n_output_nodes),
                     color=pastel((0, 1, 0)))

    total_nodes = n_input_nodes + n_hidden_nodes + n_output_nodes

    # remove edges whose weights are too low
    weight_low = 0.00001
    tmp = onp.array(weights)

    weights = jnp.where(jnp.abs(weights) < weight_low, 0, weights)

    tmp = onp.array(weights)

    edges = []
    nodes_to_remove = []
    for inode in range(total_nodes):

        n_connections = jnp.sum(weights[inode, :]) + jnp.sum(weights[:, inode])
        if n_connections == 0:
            nodes_to_remove.append(inode)
        else:
            for onode in range(total_nodes):
                if jnp.any(weights[inode, onode] != 0.0):
                    edges.append((inode, onode))

    mask = onp.array(mask).tolist()
    nodes_to_remove = [idx for idx, el in enumerate(mask) if el == 0]

    g.add_edges_from(edges, color="black")
    g.remove_nodes_from(nodes_to_remove)

    print("nodes to remove ", str(len(nodes_to_remove)))

    node_colors = [g.nodes[n]['color'] for n in g.nodes]
    edge_colors = [g.edges[e]['color'] for e in g.edges]

    node_size = 500
    edge_widths = scale_edge_widths(g, weights)

    kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust the number of clusters as needed
    cluster_labels = kmeans.fit_predict(hidden)
    cmap = plt.get_cmap('Set1')
    norm = plt.Normalize(min(cluster_labels), max(cluster_labels))
    node_colors = [cmap(norm(value)) for value in cluster_labels]

    nx.draw(g, pos=nx.fruchterman_reingold_layout(g), node_color=node_colors, node_size=node_size,
            width=[edge_widths[edge] for edge in g.edges],
            edge_color=edge_colors,
            with_labels=False)

    # Save the graph to a file (e.g., PNG, PDF, SVG, etc.)
    plt.savefig(network_filename, format='png')
    plt.clf()
    plt.close()
    num_nodes = len(g.nodes)
    num_edges = len(g.edges)

    return num_nodes, num_edges
