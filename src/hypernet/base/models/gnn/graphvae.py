from models.gnn.layers import aggregate
from src.models.gnn.base import Graph, Node, Edge, GraphModule

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn

class GraphVAE(GraphModule):
	
	"""
	Variational Graph Auto-Encoders
	paper: https://arxiv.org/pdf/1611.07308.pdf
	"""
	#-------------------------------------------------------------------
	W0: nn.Linear
	W_mu: nn.Linear
	W_sigma: nn.Linear
	#-------------------------------------------------------------------

	def __init__(self, X_features: int, Z_features: int, *, key: jax.Array):
		
		key_0, key_mu, key_sigma = jr.split(key, 3)
		
		self.W0 = nn.Linear(X_features, X_features, use_bias=False, key=key_0)
		self.W_mu = nn.Linear(X_features, Z_features, use_bias=False, key=key_sigma)
		self.W_sigma = nn.Linear(X_features, Z_features, use_bias=False, key=key_mu)

	#-------------------------------------------------------------------

	def encode(self, graph: Graph, key: jax.Array) -> Graph:
		assert graph.edges.A is not None 

		A = jnp.clip(graph.edges.A + jnp.identity(graph.edges.A.shape[0]), 0., 1.)
		h = graph.h
		x = jnn.relu(jax.vmap(self.W0)(aggregate(h, A)))
		sigma = jax.vmap(self.W_sigma)(aggregate(x, A))
		mu = jax.vmap(self.W_mu)(aggregate(x, A))

		h = jnp.concatenate([mu,sigma], axis=-1)
		nodes = graph.nodes._replace(h=h)
		return graph._replace(nodes=nodes)

	#-------------------------------------------------------------------

	def decode(self, graph: Graph, key: jax.Array) -> Graph:
		z = graph.h
		pA = jnn.sigmoid(jnp.sum(z[:,None] * z[None,...], axis=-1))
		edges = graph.edges._replace(A=pA)
		return graph._replace(edges=edges)




