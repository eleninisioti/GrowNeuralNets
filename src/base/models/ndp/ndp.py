from functools import partial
from typing import Callable, Tuple, Union, Optional, NamedTuple
from jax._src.lax.utils import _input_dtype
from jaxtyping import Float, PyTree
from src.models.base import DevelopmentalModel
from src.gnn.base import Graph, GraphModule, Node, Edge
from src.gnn.layers import GNCA, GAT, GraphMLP

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn

def get_neighbours(x, A):
	"""x is a one hot encoding of node
	A is adjecency matrix"""
	return (x[None, :] * A).sum(1)

def off_sigmoid(x, off, T):
	""""""
	return jnn.sigmoid(T*(x-off))

#=======================================================================
#=======================================================================
#=======================================================================

class NDP(DevelopmentalModel):
	
	"""
	"""
	#-------------------------------------------------------------------
	node_fn: Union[PyTree, Callable]
	edge_fn: Union[PyTree, Callable]
	div_fn: Union[PyTree, Callable]
	max_nodes: int
	init_nodes: int
	node_features: int
	edge_features: int
	#-------------------------------------------------------------------

	def __init__(self, node_fn, edge_fn, div_fn, max_nodes, init_nodes,
				 node_features, edge_features):
		
		self.node_fn = node_fn
		self.edge_fn = edge_fn
		self.div_fn = div_fn
		self.max_nodes = max_nodes
		self.init_nodes = init_nodes
		self.node_features = node_features
		self.edge_features = edge_features

	#-------------------------------------------------------------------

	def __call__(self, graph: Graph, key: jax.Array)->Graph:
		""""""
		key_node, key_growth, key_edges = jr.split(key, 3)
		graph = self.node_fn(graph, key_node)
		graph = self.add_new_nodes(graph, key_growth)
		graph = self.update_edges(graph, key_edges)

		return graph

	#-------------------------------------------------------------------

	def add_new_nodes(self, graph: Graph, key: jax.Array)->Graph:
		""""""
		assert graph.nodes.m is not None
		assert graph.edges.A is not None
		
		grow = self.div_fn(graph, key)[...,0] * graph.nodes.m
		n_grow = grow.sum()
		
		mask = graph.nodes.m
		N = mask.shape[0]
		n_alive = mask.sum()
		A = graph.edges.A
		
		new_n = n_alive+n_grow
		new_mask = jnp.arange(graph.nodes.h.shape[0]) < new_n
		xnew_mask = new_mask - mask

		# compute childs index for each parent: pc[parent] = child index
		pc = (jnp.where(grow, jnp.cumsum(grow)-1, -1) + (n_alive*grow)).astype(int)
		# Set child´s incoming connections (parent neighbors + parents)
		nA = jax.ops.segment_sum(jnp.clip(A+jnp.identity(N), 0., 1.).T, pc, N).T 
		A = jnp.where(xnew_mask[None,:], nA, A) * new_mask[None,:] * new_mask[:,None]

		nodes = graph.nodes._replace(m=new_mask.astype(float))
		edges = graph.edges._replace(A=A.astype(float))

		return graph._replace(nodes=nodes, edges=edges)

	#-------------------------------------------------------------------

	def update_edges(self, graph: Graph, key: jax.Array)->Graph:
		""""""
		edges = graph.edges
		h = graph.nodes.h
		n = h.shape[0]
		cat_rs = jnp.concatenate(
			[jnp.repeat(h[:,None], n, axis=1),
			 jnp.repeat(h[None,:], n, axis=0)],
			 axis=-1
		)
		e = jax.vmap(jax.vmap(self.edge_fn))(cat_rs) # N x N x De
		e = e * graph.edges.A[...,None] # N x N x De #type: ignore
		edges = edges._replace(e=e)
		return graph._replace(edges=edges)

	#-------------------------------------------------------------------

	def initialize(self, key: jax.Array) -> Graph:
		""""""
		nodes = Node(
			h=jnp.zeros((self.max_nodes, self.node_features)),
			m=jnp.zeros((self.max_nodes,)).at[:self.init_nodes].set(1.))
		edges = Edge(
			A=jnp.zeros((self.max_nodes, self.max_nodes)), 
		 	e=jnp.zeros((self.max_nodes, self.max_nodes, self.edge_features)))

		return Graph(nodes=nodes, edges=edges)

#=======================================================================
#=======================================================================
#=======================================================================

class ENDP(NDP):
	
	"""
	"""
	#-------------------------------------------------------------------
	# Parameters:
	death_fn: Optional[Callable]
	# Statics:
	alpha: float
	#-------------------------------------------------------------------

	def __init__(
		self, 
		max_nodes: int, 
		init_nodes: int,
		node_features: int,
		edge_features: int,
		node_fn: Optional[Callable]=None, 
		edge_fn: Optional[Callable]=None, 
		div_fn: Optional[Callable]=None, 
		death_fn: Optional[Callable]=None,
		alpha: float=.1):

		assert edge_features >= 2
		super().__init__(node_fn, edge_fn, div_fn, max_nodes, init_nodes,
						 node_features, edge_features)
		self.alpha = alpha
		self.death_fn = death_fn

	#-------------------------------------------------------------------

	def add_new_nodes(self, graph: Graph, key: jax.Array)->Graph:
		""""""
		assert graph.nodes.m is not None
		assert graph.edges.A is not None
		
		grow = self.div_fn(graph, key)[...,0] * graph.nodes.m
		n_grow = grow.sum()
		
		mask = graph.nodes.m
		N = mask.shape[0]
		n_alive = mask.sum()
		A = graph.edges.A
		
		new_n = n_alive+n_grow
		new_mask = (jnp.arange(graph.nodes.h.shape[0]) < new_n).astype(float)
		xnew_mask = new_mask - mask

		# compute childs index for each parent: pc[parent id] = child id
		pc = (jnp.where(grow, jnp.cumsum(grow)-1, -1) + (n_alive*grow)).astype(int)
		# Set child´s incoming connections (parent neighbors + parents)
		nA = jax.ops.segment_sum(jnp.identity(N), pc, N).T 
		A = jnp.where(xnew_mask[None,:], nA, A) * new_mask[None,:] * new_mask[:,None]

		nh = jax.ops.segment_sum(graph.h, pc, N)
		h = jnp.where(xnew_mask[:,None], nh, graph.h)

		nodes = graph.nodes._replace(m=new_mask.astype(float), h=h)
		edges = graph.edges._replace(A=A.astype(float))

		return graph._replace(nodes=nodes, edges=edges)

	#-------------------------------------------------------------------

	def update_edges(self, graph: Graph, key: jax.Array)->Graph:
		""""""
		assert graph.nodes.m is not None
		
		edges = graph.edges
		h = graph.nodes.h
		n = h.shape[0]
		mA = graph.nodes.m[None, :] * graph.nodes.m[:,None]
		
		pre = jnp.repeat(h[:,None], n, axis=1)
		post = jnp.repeat(h[None,:], n, axis=0)
		e = jax.vmap(jax.vmap(self.edge_fn))(graph.edges.e, pre, post) # N x N x De
		A = jnp.clip((e[...,0]>self.alpha).astype(float) * mA + graph.edges.A, 0, 1)
		e = e * A[...,None]
		edges = edges._replace(e=e, A=A)
		return graph._replace(edges=edges)

#=======================================================================
#=======================================================================
#=======================================================================

from src.gnn.layers import MPNN, EMPNN

class SimpleNDP(NDP):
	
	"""
	From: 
		Towards Self-Assembling Artificial Neural Networks through 
		Neural Developmental Programs
	paper: 
		https://arxiv.org/abs/2307.08197
	"""
	#-------------------------------------------------------------------
	weighted: bool
	#-------------------------------------------------------------------

	def __init__(self, node_features: int, msg_features: int, edge_features: int, 
				 init_nodes: int, max_nodes: int, *, key: jax.Array,
				 weighted: bool=True, pruning: bool=False, conv_steps: int=1,
				 P: float = .01, prob_activation=partial(off_sigmoid,off=1.,T=3.),
				 use_edges: bool=False):

		key_node, key_edge, key_div = jr.split(key, 3)
		if not use_edges:
			node_fn = MPNN(node_features, msg_features, node_features, key=key_node)
		else: 
			node_fn = EMPNN(node_features, edge_features, msg_features, node_features, key=key_node)
		if weighted:
			edge_fn = nn.MLP(2*node_features, edge_features, 64, 1, key=key_edge)
		else: 
			edge_fn = None
		self.weighted = weighted
		div_fn = nn.MLP(node_features, 1, 64, 1, final_activation=prob_activation,
						key=key_div)

		super().__init__(node_fn=node_fn, edge_fn=edge_fn, div_fn=div_fn, 
						 max_nodes=max_nodes, init_nodes=init_nodes, node_features=node_features,
						 edge_features=edge_features)

	#-------------------------------------------------------------------

	def update_edges(self, graph: Graph, key: jax.Array) -> Graph:
		if self.weighted:
			return super().update_edges(graph, key)
		else:
			return graph


#=======================================================================
#=======================================================================
#=======================================================================










