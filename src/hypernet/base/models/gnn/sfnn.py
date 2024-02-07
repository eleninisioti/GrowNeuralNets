from ast import NotIn
from collections import namedtuple
from typing import Callable, NamedTuple, Union, TypeAlias

from jaxtyping import Float, Int
from src.models.gnn.base import Edge, Graph, GraphModule, Node
from src.models.gnn.layers import GAT
from src.models.nn.layers import RNN, MGU

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn

RecurrentNeuralNetwork: TypeAlias = Union[nn.GRUCell, MGU, RNN]

class NodeType(NamedTuple):
	types: jax.Array

class RLData(NamedTuple):
	r: Float

class SFNN(GraphModule):
	"""
	Structurally Flexible Neural Network
	"""
	#-------------------------------------------------------------------
	node_cells: RecurrentNeuralNetwork
	node_output: nn.Sequential
	edge_cells: RecurrentNeuralNetwork
	#-------------------------------------------------------------------

	def __init__(self, hidden_dims: int, msg_dims: int, n_types: int, cell_type: str="gru", activation: Callable=jnn.relu, *, key: jax.Array):

		key_ncells, key_nout, key_ecells = jr.split(key, 3)
		
		def init_node_cell(key):
			if cell_type == "gru":
				return nn.GRUCell(msg_dims, hidden_dims, key=key)
			elif cell_type == "mgu":
				return MGU(msg_dims, hidden_dims, key=key)
			elif cell_type == "rnn":
				return RNN(hidden_dims, msg_dims, key=key)
			else : 
				raise ValueError(f"{cell_type} is not a known or managed cell type")

		self.node_cells = jax.vmap(init_node_cell)(jr.split(key_ncells, n_types))
		self.node_output = nn.Sequential(
			[
				nn.Linear(hidden_dims, msg_dims, key=key_nout),
				nn.Lambda(activation)
			]
		)

		def init_edge_cell(key):
			in_dims = 2*msg_dims+1
			out_dims=msg_dims
			if cell_type == "gru":
				return nn.GRUCell(in_dims, out_dims, key=key)
			elif cell_type == "mgu":
				return MGU(in_dims, out_dims, key=key)
			elif cell_type == "rnn":
				return RNN(in_dims, out_dims, key=key)
			else : 
				raise ValueError(f"{cell_type} is not a known or managed cell type")

		self.edge_cells = jax.vmap(init_edge_cell)(jr.split(key_ecells, n_types))

	#-------------------------------------------------------------------

	def apply_adj(self, graph: Graph, key: jax.Array) -> Graph:
		"""
		"""
		assert graph.pholder is not None
		assert graph.nodes.pholder is not None
		assert graph.A is not None
		assert graph.edges.e is not None

		h = graph.h 	  # N x H
		A = graph.A 	  # N x N
		e = graph.edges.e # N x N x M
		N = h.shape[0]
		node_types = graph.nodes.pholder.types # N: int
		edge_types = jnp.repeat(node_types[:,None], N, axis=1) # N x N: int

		# 1. compute and aggregate signals
		y = jax.vmap(self.node_output)(h) 		  # N x M
		m = y[:, None, :] * (e * A[...,None])  	  # N x N x M
		x = m.sum(1) 		  			 		  # N x M

		# 2. Update node states
		h = jax.vmap(self._apply_node_cell)(x, h, node_types) # N x H

		# 3. Update edges states
		yiyjr = jnp.concatenate(	# N x N x 2M
			[
				jnp.repeat(y[:,None], N, axis=1),
			 	jnp.repeat(y[None,:], N, axis=0),
			 	jnp.ones((N,N,1)) * graph.pholder.r
		 	],
			 axis=-1
		)
		e = jax.vmap(jax.vmap(self._apply_edge_cell))(yiyjr, e, edge_types) # N x N x M

		nodes = graph.nodes._replace(h=h)
		edges = graph.edges._replace(e=e)
		return graph._replace(nodes=nodes, edges=edges)

	#-------------------------------------------------------------------

	def _apply_node_cell(self, x: jax.Array, h: jax.Array, typ: Int):
		cell = jax.tree_map(lambda x: x[typ] if eqx.is_array(x) else x, self.node_cells)
		return cell(x, h)

	def _apply_edge_cell(self, x: jax.Array, e: jax.Array, typ: Int):
		cell = jax.tree_map(lambda x: x[typ] if eqx.is_array(x) else x, self.edge_cells)
		return cell(x, e)

	#-------------------------------------------------------------------


#=======================================================================
#=======================================================================
#=======================================================================

class GATSFNN(GraphModule):
	
	"""
	"""
	#-------------------------------------------------------------------
	# Parameters:
	gat: GAT
	cell: RecurrentNeuralNetwork
	# Statics:
	
	#-------------------------------------------------------------------

	def __init__(self, node_features: int, msg_features: int, edge_features: int, cell: str="gru", 
				 gat_heads: int=4, gat_width: int=64, gat_depth: int=2, *, key: jax.Array):
		
		key_gat, key_cell = jr.split(key)
		self.gat = GAT(node_features, msg_features, gat_heads, edge_features=edge_features,
					   att_depth=gat_depth, att_width=gat_width, key=key_gat)
		if cell == "gru":
			self.cell = nn.GRUCell(msg_features, node_features, key=key_cell)
		else :
			raise NotImplementedError(f"{cell} not impl")


	#-------------------------------------------------------------------

	def apply_adj(self, graph: Graph, key: jax.Array) -> Graph:
		msg = self.gat(graph, key).h
		h = jax.vmap(self.cell)(msg, h)
		return eqx.tree_at(lambda G: G.nodes.h, graph, h)

	#-------------------------------------------------------------------

	def initialize(self, key: jax.Array):
		nodes = Node(h=jnp.zeros())
		pass

#=======================================================================

if __name__ == '__main__':
	import timeit

	key = jr.PRNGKey(1)
	kt, kh, ke, kA, km, kc = jr.split(key, 6)

	N = 1024
	Dh = 4
	Dm = 2
	n_types = 16

	typs = NodeType(types=jr.randint(kt, (N,), 0, n_types))
	h = jr.normal(kh, (N, Dh))
	e = jr.normal(ke, (N, N, Dm))
	A = jnp.clip(jr.randint(kA, (N,N), 0, 2)-jnp.identity(N), 0, 1)
	infos = RLData(r=.2)

	G = Graph(
		nodes = Node(
			h=h,
			pholder=typs
		),
		edges = Edge(
			A=A,
			e=e
		),
		pholder=infos
	)

	M = eqx.filter_jit(SFNN(Dh, Dm, n_types, key=km))
	
	time = timeit.timeit(lambda: M(G, kc), number=128)
	print(time)









