from ast import Call
from functools import partial
from typing import Callable, Union, Optional
from jaxtyping import Float, PyTree
from src.NDP.base.models.base import DevelopmentalModel
from src.NDP.base.models.gnn.base import Graph, Node, Edge
from src.NDP.base.models.gnn.layers import GNCA, GAT, GraphMLP
import numpy as onp
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnni
import equinox as eqx
import equinox.nn as nn
from flax import linen


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
	node_features_intrinsic: int
	inhibit_mitosis: bool
	inhibit_weights: bool
	inhibit_for: int = 10
	intrinsic: bool = False,
	#-------------------------------------------------------------------

	def __init__(self, node_fn, edge_fn, div_fn, max_nodes, init_nodes,
				 node_features, edge_features, node_features_intrinsic, inhibit_mitosis, inhibit_weights,inhibit_for,intrinsic):
		
		self.node_fn = node_fn
		self.edge_fn = edge_fn
		self.div_fn = div_fn
		self.max_nodes = max_nodes
		self.init_nodes = init_nodes
		self.node_features = node_features
		self.edge_features = edge_features
		self.node_features_intrinsic = node_features_intrinsic
		self.intrinsic = intrinsic
		self.inhibit_for = inhibit_for
		self.inhibit_mitosis = inhibit_mitosis
		self.inhibit_weights =inhibit_weights
		self.intrinsic = intrinsic

	#-------------------------------------------------------------------

	def __call__(self, graph: Graph, key: jax.Array, counter: int)->Graph:
		""""""
		key_node, key_growth, key_edges = jr.split(key, 3)

		graph = self.node_fn(graph, key_node, counter=counter)
		graph = self.add_new_nodes(graph, key_growth, counter=counter)

		e = self.update_edges(graph, key_edges, counter=counter)
		#e = jnp.where(counter==15, e, jnp.zeros(e.shape))

		mA = graph.nodes.m[None, :] * graph.nodes.m[:,None]

		A = jnp.clip((e[..., 0] > self.alpha).astype(float) * mA + graph.edges.A, 0, 1)
		e = e * A[..., None]
		edges = graph.edges._replace(e=e, A=A)
		graph = graph._replace(edges=edges)
		return graph



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
		node_features_intrinsic: int=0,
		inhibit_mitosis: bool=False,
		inhibit_weights: bool=False,
		alpha: float=-10,
	    inhibit_for: int=10,
		intrinsic: bool=False,
	):

		#assert edge_features >= 2
		super().__init__(node_fn=node_fn,
						 edge_fn=edge_fn,
						 div_fn=div_fn,
						 max_nodes=max_nodes,
						 init_nodes=init_nodes,
						 node_features_intrinsic=node_features_intrinsic,
						 node_features=node_features,
						 edge_features=edge_features,
						 inhibit_mitosis=inhibit_mitosis,
						 inhibit_weights=inhibit_weights,

						 inhibit_for=inhibit_for,
						 intrinsic=intrinsic)

		self.alpha = alpha
		self.death_fn = death_fn

	#-------------------------------------------------------------------

	def add_new_nodes(self, graph: Graph, key: jax.Array, counter: int)->Graph:
		""""""
		assert graph.nodes.m is not None
		assert graph.edges.A is not None
		
		grow = self.div_fn(graph, key)[...,0] * graph.nodes.m
		grow_before = jnp.zeros(grow.shape)


		@jax.jit
		def lateral_inhibition(hidden_before, hidden_after, index, inhibitions, connections):
			hidden_inhibited = jnp.copy(hidden_before)
			hidden_inhibited = jnp.where(inhibitions > 0, hidden_inhibited, hidden_after)
			other_inhibitions = jnp.where(connections, 1, 0)
			other_inhibitions = jnp.where(inhibitions,jnp.zeros(other_inhibitions.shape), other_inhibitions)
			other_inhibitions = other_inhibitions.at[index].set(0)

			return hidden_inhibited, other_inhibitions


		if self.inhibit_mitosis:
			inhib_counter = self.inhibit_for
			indexes = jnp.arange(jnp.shape(grow_before)[0])
			inhibitions = graph.nodes.inhibited_node
			connections = graph.edges.A

			for node in indexes:
				next_key, key = jax.random.split(key)
				grow_node, inhibitions_node = lateral_inhibition(grow_before[node, ...], grow[node, ...], node, inhibitions[node, ...],
													connections[node, ...])
				prob_noinhib = 1 / (counter + 1)
				random_array = jax.random.choice(next_key, 2, shape=inhibitions.shape,
												 p=jnp.array([prob_noinhib, 1 - prob_noinhib]))
				inhibitions_node = jnp.where(random_array, 0, inhibitions_node)
				#inhibitions_node =jnp.zeros(inhibitions_node.shape)
				#inhibitions_node.at[(node+2)%indexes.shape[0]].set(1)
				inhibitions = jnp.add(inhibitions, inhibitions_node)

				grow = grow.at[node, ...].set(grow_node)

			new_inhibitions = inhibitions

			#grow, new_inhibitions = jax.vmap(lateral_inhibition)(grow_before, grow, indexes, inhibitions, connections)
			#grow = global_inhibition(grow, key)
			new_inhibitions = jnp.where(inhibitions, inhib_counter, graph.nodes.inhibited_hidden)
			new_inhibitions = jnp.where(new_inhibitions, new_inhibitions - 1, new_inhibitions)
			#summed = jnp.sum(new_inhibitions, axis=1)
			#inhibited = jnp.where(summed, 1, 0)
			#new_inhibitions = jnp.where(inhibited, inhib_counter, inhibitions)
			#new_inhibitions = jnp.where(new_inhibitions, new_inhibitions - 1, new_inhibitions)
			#grow = jnp.where(new_inhibitions, grow_before, grow)

			graph = graph._replace(nodes=graph.nodes._replace(inhibited_node=new_inhibitions))
			#first_grow = grow
			#first_grow = first_grow.at[4:].set(0)
			grow = jnp.where(counter == 1, jnp.zeros(grow.shape), grow)
			grow = jnp.where(counter == 2, jnp.zeros(grow.shape), grow)

		grow = jnp.where(grow>0,1,0)
		#grow = jnp.zeros(grow.shape)
		#grow = grow.at[0:4].set(1)
		mask = graph.nodes.m
		#grow =jnp.where(mask, 0, grow)
		n_grow = grow.sum()
		N = mask.shape[0]
		n_alive = mask.sum()
		A = graph.edges.A
		new_n = n_alive+n_grow
		new_mask = (jnp.arange(graph.nodes.h_learned.shape[0]) < new_n).astype(float)
		xnew_mask = new_mask - mask

		# compute childs index for each parent: pc[parent id] = child id
		pc = (jnp.where(grow, jnp.cumsum(grow)-1, -1) + (n_alive*grow)).astype(int)
		# Set child´s incoming connections (parent neighbors + parents)
		nA = jax.ops.segment_sum(jnp.identity(N), pc, N).T 
		A = jnp.where(xnew_mask[None,:], nA, A) * new_mask[None,:] * new_mask[:,None]

		nh = jax.ops.segment_sum(graph.h, pc, N)
		nh = jr.normal(key, nh.shape)
		h = jnp.where(xnew_mask[:,None], nh, graph.h)

		h_intrinsic = graph.nodes.h_intrinsic
		max_nodes = h_intrinsic.shape[0]
		init_nodes = jnp.int32(n_alive)
		#init_nodes = jnp.int32(jr.randint(key, (1,), 1, self.max_nodes ))

		current_state = jnp.add(jnp.argmax(h_intrinsic, axis=1), jnp.zeros((max_nodes,))).astype(jnp.int32)
		mutated_intrinsic = jnp.zeros(h_intrinsic.shape)
		mutation = jr.normal(key, (1,), jnp.float32)
		mutate_onehot= jr.randint(key, (1,), 1, self.max_nodes, jnp.int32)
		#mutated_intrinsic = mutated_intrinsic.at[jnp.arange(max_nodes), (current_state + init_nodes)%self.max_nodes].set(1+ mutation)
		#mutated_intrinsic = mutated_intrinsic.at[jnp.arange(max_nodes), (current_state + mutate_onehot)%(3*self.max_nodes)].set(1)
		mutated_intrinsic = mutated_intrinsic.at[jnp.arange(max_nodes), (current_state  )%(self.max_nodes)].set(1)

		indexes = jnp.arange(max_nodes)
		kids = jnp.where(pc > 0, indexes, -1)
		new_values = mutated_intrinsic[kids]
		mutated_intrinsic = mutated_intrinsic.at[pc].set(new_values)

		#mutated_intrinsic = mutated_intrinsic.at[jnp.int32(n_alive + n_grow):].set(0)

		#x_vals = jnp.arange(0, self.max_nodes)
		#y_vals = jnp.arange(0, self.max_nodes)
		#grid = jnp.meshgrid(x_vals, y_vals, indexing="xy")
		#mutated_intrinsic = jnp.where(grid[1] > jnp.int32(n_alive + n_grow), jnp.zeros(mutated_intrinsic), mutated_intrinsic)

		h_intrinsic = jnp.where(xnew_mask[:, None], mutated_intrinsic, h_intrinsic)

		#h_intrinsic = jr.normal(key, h_intrinsic.shape)
		#h_intrinsic = h_intrinsic


		nodes = graph.nodes._replace(m=new_mask.astype(float), h_learned=h, h_intrinsic=h_intrinsic)
		edges = graph.edges._replace(A=A.astype(float))

		return graph._replace(nodes=nodes, edges=edges)

	#-------------------------------------------------------------------

	def update_edges(self, graph: Graph, key: jax.Array, counter: int)->Graph:
		""""""
		assert graph.nodes.m is not None
		
		edges = graph.edges
		h = graph.nodes.h_learned
		num_nodes = h.shape[0]

		if self.intrinsic:
			h = jnp.concatenate([h, graph.nodes.h_intrinsic], axis=1)

		n = h.shape[0]
		temp1 = graph.nodes.m[None, :]
		temp2= graph.nodes.m[:,None]
		mA = graph.nodes.m[None, :] * graph.nodes.m[:,None]
		
		pre = jnp.repeat(h[:,None], n, axis=1)
		post = jnp.repeat(h[None,:], n, axis=0)
		e = jax.vmap(jax.vmap(self.edge_fn))(graph.edges.e, pre, post) # N x N x De

		#e = e.at[5:8,:,0].set(0)
		#A = jnp.clip((e[...,0]>self.alpha).astype(float) * mA + graph.edges.A, 0, 1)
		#e = e * A[...,None]
		#A = jnp.ones(jnp.shape(A))
		edges_before = edges.e
		edges_after = e

		def lateral_inhibition(key, hidden_before, hidden_after, index, inhibitions, connections):
			hidden_inhibited = jnp.copy(hidden_before)
			total_nodes = jnp.count_nonzero(graph.nodes.m)

			candidate_2 = (jax.random.randint(key, minval=0, maxval=total_nodes, shape=(1,)))[0]
			hidden_inhibited = jax.lax.cond(inhibitions>0,
												lambda x: x,
												lambda x: hidden_inhibited.at[candidate_2,:].set(
													hidden_after[candidate_2,...]),
												hidden_inhibited)

			other_inhibitions = jnp.where(connections, 1, 0)
			other_inhibitions = jnp.where(inhibitions > 0, jnp.zeros(other_inhibitions.shape), other_inhibitions)

			other_inhibitions = other_inhibitions.at[index].set(0)

			return hidden_inhibited, other_inhibitions

		if self.inhibit_weights:
			inhib_counter = self.inhibit_for
			indexes = jnp.arange(edges_before.shape[0])
			inhibitions = graph.nodes.inhibited_edge
			connections = graph.edges.A
			keys = jax.random.split(key, jnp.shape(inhibitions)[0])

			nodes_mask = graph.nodes.m
			edges_mask = jnp.outer(nodes_mask, nodes_mask)
			# edges_mask = jnp.multiply(edges_mask, (1 - jnp.eye(edges_mask.shape[0])))
			edges_mask = jnp.expand_dims(edges_mask, axis=-1)
			edges_after = jnp.where(edges_mask, edges_after, jnp.zeros(edges_after.shape))
			edges_before = jnp.where(edges_mask, edges_before, jnp.zeros(edges_after.shape))

			for node in indexes:
				next_key, key = jax.random.split(key)
				e_node, inhibitions_node = lateral_inhibition(keys[node, ...], edges_before[node, ...],
															  edges_after[node, ...], node, inhibitions[node, ...],
															  connections[node, ...])
				prob_noinhib = 1 / (counter + 1)
				random_array = jax.random.choice(next_key, 2, shape=inhibitions.shape,
												 p=jnp.array([prob_noinhib, 1 - prob_noinhib]))
				inhibitions_node = jnp.where(random_array, 0, inhibitions_node)
				# inhibitions_node =jnp.zeros(inhibitions_node.shape)

				inhibitions = jnp.add(inhibitions, inhibitions_node)

				e = e.at[node, ...].set(e_node)

			# new_inhibitions = inhibitions
			# e, new_inhibitions = jax.vmap(lateral_inhibition)(keys, edges_before, edges_after,  indexes, inhibitions, connections)
			# e = global_inhibition(edges_before, e, mA, graph.nodes.m, key)
			new_inhibitions = jnp.where(inhibitions, inhib_counter, graph.nodes.inhibited_hidden)
			new_inhibitions = jnp.where(new_inhibitions, new_inhibitions - 1, new_inhibitions)

			# e = edges_after
			##summed = jnp.sum(new_inhibitions, axis=1)
			# inhibited = jnp.where(summed, 1, 0)
			# new_inhibitions = jnp.where(inhibited, inhib_counter, inhibitions)
			# new_inhibitions = jnp.where(new_inhibitions, new_inhibitions - 1, new_inhibitions)
			# temp_inhibitions = jnp.outer(new_inhibitions, new_inhibitions)
			# temp_inhibitions = jnp.expand_dims(temp_inhibitions, axis=-1)

			# temp_inhibitions = jnp.tile(temp_inhibitions, reps=(1, 1, e.shape[2]))

			# e = jnp.where(temp_inhibitions, edges_before, e)

			graph = graph._replace(nodes=graph.nodes._replace(inhibited_edge=new_inhibitions))

			# lr = 0.01
			# e = edges_before + lr*e

			# e = jnp.where(counter==1, edges_before, e)
			A = jnp.clip((e[..., 0] > self.alpha).astype(float) * mA + graph.edges.A, 0, 1)
			e = e * A[..., None]

			graph = graph._replace(edges=graph.edges._replace(e=e, A=A))

		else:
			A = jnp.clip((e[..., 0] > self.alpha).astype(float) * mA + graph.edges.A, 0, 1)
			e = e * A[..., None]
			edges = edges._replace(e=e, A=A)
			graph = graph._replace(edges=edges)

		return e

#=======================================================================
#=======================================================================
#=======================================================================

from src.NDP.base.models.gnn.layers import MPNN, EMPNN

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
from src.NDP.base.models.gnn.base import GraphModule
from src.NDP.base.models.gnn.layers import aggregate

class RecurrentGNCA(GraphModule):
	"""
	"""
	#-------------------------------------------------------------------
	pre_mlp: nn.MLP
	cell: nn.GRUCell
	#-------------------------------------------------------------------

	def __init__(self, node_features: int, msg_features: int, *, key: jax.Array):
		key_pre, key_cell = jr.split(key)
		
		self.pre_mlp = nn.MLP(node_features, msg_features, 32, 2, key=key_pre)
		self.cell = nn.GRUCell(msg_features, node_features, key=key_cell)

	#-------------------------------------------------------------------

	def apply_adj(self, graph: Graph, key: jax.Array) -> Graph:
		assert graph.edges.A is not None

		m = jax.vmap(self.pre_mlp)(graph.nodes.h)
		x = aggregate(m, graph.edges.A)
		h = jax.vmap(self.cell)(x, graph.nodes.h)

		nodes = graph.nodes._replace(h=h)
		return graph._replace(nodes=nodes)

class RecurrentNDP(NDP):
	
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
				 P: float = .01, prob_activation=partial(off_sigmoid,off=1.,T=3.)):

		key_node, key_edge, key_div = jr.split(key, 3)
		
		node_fn = RecurrentGNCA(node_features, msg_features, key=key_node)
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

class AttNDP(NDP):
	
	"""
	"""
	#-------------------------------------------------------------------
	weighted: bool
	#-------------------------------------------------------------------

	def __init__(self, max_nodes: int, init_nodes: int, node_features: int, 
			     edge_features: int, att_heads: int, *, key: jax.Array, weighted:bool=True, 
			     pruning: bool=False, P: Optional[float]=None, prob_activation=partial(off_sigmoid,off=1.,T=3.),
			     use_edges: bool=False):
		
		key_node, key_edge, key_div = jr.split(key, 3)
		if not use_edges:
			node_fn = GAT(node_features, node_features, n_heads=att_heads, att_depth=2, att_width=32, key=key_node)
		else:
			node_fn = GAT(node_features, node_features, n_heads=att_heads, att_depth=2, att_width=32, key=key_node, 
						  use_edges=True, edge_features=edge_features)
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

if __name__ == '__main__':
	key = jr.PRNGKey(101)
	F = 8
	node_fn = GraphMLP(F, F, 16, 1, key=key) #type: ignore
	edge_fn = lambda pr, po, e: jnp.ones((3,))
	div_fn = lambda G, k: (jr.uniform(k, (G.h.shape[0],))<jnp.array([.8])).astype(float)

	ndp = ENDP(node_fn, edge_fn, div_fn, 10, 1, F, 3, alpha=.01)
	G = ndp.initialize(key)
	age = G.nodes.m
	for i in range(2):
		key, skey = jr.split(key)
		G = ndp(G, skey)
		age = age + G.nodes.m
		c = age[G.nodes.m.astype(bool)]
		#viz_graph(G, node_color=c, cmap="rainbow")
		#print(G.edges.A)
		#print(G.nodes.m)




