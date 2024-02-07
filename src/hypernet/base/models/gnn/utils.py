from typing import Callable
from jaxtyping import Array, Float
from AI_frameworks.NDP.models.gnn.base import Graph, Node, Edge
from AI_frameworks.NDP.models.nn.layers import RNN

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn
import networkx as nx
import matplotlib.pyplot as plt


def in_degrees(A: Float[Array, "N N"])->Float[Array, "N"]:
	return A.sum(0)

def out_degrees(A: Float[Array, "N N"])->Float[Array, "N"]:
	return A.sum(1)

def degrees(A: Float[Array, "N N"])-> Float[Array, "N"]:
	return out_degrees(A) + in_degrees(A)

def count_k_cycles(graph: Graph, k):
	assert graph.edges.A is not None
	A = graph.edges.A
	# Compute the kth power of the adjacency matrix
	A = jax.lax.fori_loop(0, k-1, lambda i, Ai: jnp.matmul(Ai, A), A)
	# Count the number of k-cycles for each node
	cycle_counts = jnp.diag(A)
	return cycle_counts

def count_total_k_cycles(A, k):
	# Compute the kth power of the adjacency matrix
	A = jax.lax.fori_loop(0, k-1, lambda i, Ai: jnp.matmul(Ai, A), A)

	# The number of k-cycles in the whole graph is half the trace of Ak
	total_cycles = jnp.trace(A) // 2  # // 2 to avoid double counting
	return total_cycles



def to_networkx(graph: Graph):
	if graph.edges.A is not None:
		if graph.nodes.m is not None:
			m = graph.nodes.m.astype(bool)
			A = graph.edges.A[m][:, m].astype(int)
			return nx.from_numpy_array(A, create_using=nx.DiGraph)
		else: 
			A = graph.edges.A.astype(int)
			return nx.from_numpy_array(A)
	else:
		raise NotImplementedError

def viz_graph(graph: Graph, file, **kwargs):
	nxg = to_networkx(graph)
	nx.draw_kamada_kawai(nxg, node_size=20, **kwargs)
	plt.savefig(file)
	plt.clf()




