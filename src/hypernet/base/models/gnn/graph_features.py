from src.models.gnn.base import GraphModule, Graph, Node, Edge

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
from typing import Callable, Optional

class CycleCountInjection(GraphModule):
	"""
	"""
	#-------------------------------------------------------------------
	ks: jax.Array
	k_max: int
	mode: str
	#-------------------------------------------------------------------

	def __init__(self, ks: list[int]=[2,3,4,5,6], mode: str="cat"):
		self.ks = jnp.array(ks, dtype=int)
		self.k_max = max(ks)
		self.mode = mode

	#-------------------------------------------------------------------

	def apply_adj(self, graph: Graph, key: jax.Array) -> Graph:
		assert graph.edges.A is not None
		ccount = self.count_cycles(graph.edges.A).astype(float)
		if self.mode == "cat":
			h = jnp.concatenate([graph.h, ccount], axis=-1)
		else: 
			h = ccount
		nodes = graph.nodes._replace(h=h)
		return graph._replace(nodes=nodes)

	#-------------------------------------------------------------------

	def count_cycles(self, A: jax.Array):
		# Compute the kth power of the adjacency matrix
		def scan_pow(Ai, i):
			Aip = jnp.matmul(Ai, A)
			return Aip, Aip
		_, As = jax.lax.scan(scan_pow, A, jnp.arange(self.k_max-1)) 
		# Count the number of k-cycles for each node
		cycle_counts = jax.vmap(jnp.diag)(As[self.ks]).T # K x N
		return cycle_counts


def simple_rnn(x: jax.Array, h: jax.Array, w: jax.Array, inp_mode: str="set"):
	d = x.shape[-1]
	h = h.at[:d].set(x) if inp_mode=="set" else h.at[:d].add(x)
	h = jnn.tanh(h@w)
	return h

class GraphFeatures(GraphModule):

	#-------------------------------------------------------------------
	features: list[str]
	in_dims: Optional[int]
	rnn_iters: Optional[int]
	rnn_model: Callable
	#-------------------------------------------------------------------

	def __init__(self, features: list[str], in_dims: Optional[int]=None, 
				 rnn_iters: Optional[int]=None, rnn_model: Callable=simple_rnn):
		
		assert (in_dims is not None and rnn_iters is not None) or "dynamical" not in features
		self.features=features
		self.in_dims = in_dims
		self.rnn_iters = rnn_iters
		self.rnn_model = rnn_model

	#-------------------------------------------------------------------

	def apply_adj(self, graph, key):
		""""""
		assert graph.edges.A is not None
		h = graph.h
		A = graph.edges.A
		N = h.shape[0]
		features = jnp.zeros((N, 1))

		if "degree" in self.features:
			in_degree = A.sum(0) # N
			out_degree = A.sum(1) # N
			features = jnp.concatenate([features, in_degree[...,None], out_degree[...,None]], axis=-1)

		if "node_age" in self.features:
			assert graph.nodes.pholder is not None and hasattr(graph.nodes.pholder, "age")
			age = graph.nodes.pholder.age
			features = jnp.concatenate([features, age[:,None]], axis=-1)

		if "time" in self.features:
			assert graph.pholder is not None and hasattr(graph.pholder, "time")
			time = jnp.ones((N, 1)) * graph.pholder.time
			features = jnp.concatenate([features, time], axis=-1)

		if "cycles" in self.features:
			cycle_counter = CycleCountInjection()
			cycles = cycle_counter.count_cycles(A)
			features = jnp.concatenate([features, cycles], axis=-1)

		if "dynamical" in self.features:
			assert self.in_dims is not None and self.rnn_iters is not None
			assert graph.edges.e is not None
			m = graph.nodes.m if graph.nodes.m is not None else jnp.ones((N,))
			h = jnp.zeros_like(m)
			w = graph.edges.e[...,1] * graph.edges.A
			s = jnp.zeros(h.shape[:1])
			xs = jr.normal(key, (self.rnn_iters, self.in_dims)).at[1:, :]
			s, ss = jax.lax.scan(
				lambda s, x: (self.rnn_model(x, s, w),)*2, #type: ignore
				s, xs
			)
			features = jnp.concatenate([features, ss], axis=-1)

		h = jnp.concatenate([h, features[:, 1:]], axis=-1)
		nodes = graph.nodes._replace(h=h)
		return graph._replace(nodes=nodes)

	#-------------------------------------------------------------------

	def n_features(self):
		mapping = {"degree": 2, "node_age": 1, "time": 1, "cycles": 5, "dynamical": self.rnn_iters}
		n = 0
		for f in self.features:
			n = n + mapping[f]
		return n