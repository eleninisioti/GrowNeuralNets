

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn
from typing import Callable, Iterable, NamedTuple, Optional, Type, Union, Any
from jaxtyping import Float, Int, PyTree, Array
from flax import linen
from src.hypernet.base.models.gnn.base import (Graph, Node, Edge, GraphModule)
#================================================================================================================

def aggregate(x: Float[Array, "N Dx"], A: Float[Array, "N N"]):
	return jnp.dot(A.T, x)

def meanAggregate(x: Float[Array, "N Dx"], A: Float[Array, "N N"]):
	A = A / A.sum(0, keepdims=True)
	return aggregate(x, A)

#=================================================================================================================
#=================================================================================================================
#=================================================================================================================


class GraphMLP(nn.MLP):

	#-------------------------------------------------------------------

	def __call__(self, graph: Graph, *args, **kwargs)->Graph:
		h = jax.vmap(super().__call__)(graph.nodes.h)
		return graph._replace(nodes=graph.nodes._replace(h=h)) # pyright: ignore[reportGeneralTypeIssues]


#=================================================================================================================
#=================================================================================================================
#=================================================================================================================


class MPNN(GraphModule):
	"""
	"""
	#-------------------------------------------------------------------
	msg_fn: Callable
	node_fn: Callable
	aggr_fn: Callable
	activation: Callable
	update_mode: str
	concat_hm: bool
	#-------------------------------------------------------------------

	def __init__(self, in_features: int, msg_features: int, out_features: int, msg_fn: Optional[Callable]=None, 
				 aggr_fn: Callable=aggregate, activation: Callable=jnn.tanh, update_mode: str="add_and_norm", *, key: jax.Array,
				 concat_hm: bool=False):
		
		key_update, key_msg = jr.split(key)
		if msg_fn is None:
			self.msg_fn = nn.Linear(in_features, msg_features, key=key_msg)
		else:
			self.msg_fn = msg_fn
		self.node_fn = nn.Linear(msg_features+in_features if concat_hm else msg_features, out_features, key=key_update)
		self.aggr_fn = aggr_fn
		self.activation = activation
		self.update_mode = update_mode
		self.concat_hm = concat_hm

	#-------------------------------------------------------------------

	def apply_adj(self, graph: Graph, key: jax.Array) -> Graph:
		m = self.activation(jax.vmap(self.msg_fn)(graph.h))
		x = self.aggr_fn(m, graph.edges.A)
		if self.concat_hm:
			x = jnp.concatenate([x,graph.h],axis=-1)
		h = jax.vmap(self.node_fn)(x)
		if self.update_mode=="add_and_norm":
			if h.shape[-1] < graph.h.shape[-1]:
				h0 = graph.h[:, :h.shape[-1]]
			else:
				h0 = graph.h
			h = h + h0
			h = h / (jnp.linalg.norm(h, axis=-1, keepdims=True)+1e-8)
		elif self.update_mode=="add":
			h = h + graph.h
		nodes = graph.nodes._replace(h=h)
		return graph._replace(nodes=nodes)

class RMPNN(GraphModule):
	"""
	"""
	#-------------------------------------------------------------------
	msg_fn: Callable
	node_fn: Callable
	aggr_fn: Callable
	activation: Callable
	#-------------------------------------------------------------------

	def __init__(self, node_features: int, msg_features: int, msg_fn: Optional[Callable]=None, 
				 aggr_fn: Callable=aggregate, activation: Callable=jnn.tanh, *, cell: Type=nn.GRUCell,
				 key: jax.Array):
		
		key_update, key_msg = jr.split(key)
		if msg_fn is None:
			self.msg_fn = nn.Linear(node_features, msg_features, key=key_msg)
		else:
			self.msg_fn = msg_fn
		self.node_fn = cell(msg_features, node_features, key=key_update)
		self.aggr_fn = aggr_fn
		self.activation = activation

	#-------------------------------------------------------------------

	def apply_adj(self, graph: Graph, key: jax.Array) -> Graph:
		m = self.activation(jax.vmap(self.msg_fn)(graph.h))
		x = self.aggr_fn(m, graph.edges.A)
		h = jax.vmap(self.node_fn)(x, graph.h)
		nodes = graph.nodes._replace(h=h)
		return graph._replace(nodes=nodes)


class EMPNN(GraphModule):
	"""
	"""
	#-------------------------------------------------------------------
	msg_fn: Callable
	node_fn: Callable
	aggr_fn: Callable
	activation: Callable
	update_mode: str
	#-------------------------------------------------------------------

	def __init__(self, in_features: int, edge_features: int, msg_features: int, out_features: int, msg_fn: Optional[Callable]=None, 
				 aggr_fn: Callable=aggregate, activation: Callable=jnn.tanh, update_mode: str="add_and_norm", 
				 *, key: jax.Array):
		assert update_mode in ["add", "add_and_norm", "", None]
		key_update, key_msg = jr.split(key)
		if msg_fn is None:
			self.msg_fn = nn.Linear(in_features+edge_features, msg_features, key=key_msg)
		else:
			self.msg_fn = msg_fn
		self.node_fn = nn.Linear(msg_features, out_features, key=key_update)
		self.aggr_fn = aggr_fn
		self.activation = activation
		self.update_mode = update_mode

	#-------------------------------------------------------------------

	def apply_adj(self, graph: Graph, key: jax.Array) -> Graph:
		assert graph.edges.e is not None
		assert graph.edges.A is not None

		h = graph.h
		e = graph.edges.e
		he = jnp.concatenate(	# N x N x H+E
			[
				h[:,None]-h[None,:],
			 	e
		 	],
			 axis=-1
		)
		A = graph.edges.A
		m = self.activation(jax.vmap(jax.vmap(self.msg_fn))(he) * A[...,None])
		m = m.sum(0)
		x = self.aggr_fn(m, graph.edges.A)
		h = jax.vmap(self.node_fn)(x)
		if self.update_mode=="add_and_norm":
			if h.shape[-1] < graph.h.shape[-1]:
				h0 = graph.h[:, :h.shape[-1]]
			else:
				h0 = graph.h
			h = h + h0
			h = h / (jnp.linalg.norm(h, axis=-1, keepdims=True)+1e-8)
		elif self.update_mode=="add":
			h = h + graph.h
		nodes = graph.nodes._replace(h=h)
		return graph._replace(nodes=nodes)




#=================================================================================================================
#=================================================================================================================
#=================================================================================================================

class GNCA(GraphModule):
	"""
	"""
	#-------------------------------------------------------------------
	pre_mlp: nn.MLP
	post_mlp: nn.MLP
	#-------------------------------------------------------------------

	def __init__(self, node_features: int, msg_features: int, *, key: jax.Array):
		key_pre, key_post = jr.split(key)
		
		self.pre_mlp = nn.MLP(node_features, msg_features, 32, 1, key=key_pre)
		self.post_mlp = nn.MLP(msg_features, node_features, 32, 1, key=key_post)

	#-------------------------------------------------------------------

	def apply_adj(self, graph: Graph, key: jax.Array) -> Graph:
		assert graph.edges.A is not None

		m = jax.vmap(self.pre_mlp)(graph.nodes.h)
		x = aggregate(m, graph.edges.A)
		h = jax.vmap(self.post_mlp)(x)

		nodes = graph.nodes._replace(h=h)
		return graph._replace(nodes=nodes)

#=================================================================================================================
#=================================================================================================================
#=================================================================================================================

class GIN(GraphModule):
	
	"""
	GRAPH ISOMORPHISM NETWORK 
	paper: HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (https://arxiv.org/pdf/1810.00826.pdf)
	"""
	#-------------------------------------------------------------------
	f: nn.MLP
	eps: Float
	#-------------------------------------------------------------------

	def __init__(self, in_features: int, out_features: int, depth: int=2, 
				 width: int=128, learn_eps: bool=False, *, key: jax.Array):
		
		self.f = nn.MLP(in_features, out_features, width, depth, key=key)
		self.eps = jnp.zeros(()) if learn_eps else 0.

	#-------------------------------------------------------------------

	def apply_adj(self, graph: Graph, key: jax.Array) -> Graph:
		assert graph.edges.A is not None
		h, A = graph.nodes.h, graph.edges.A
		m = aggregate(h, A)
		h = jax.vmap(self.f)((1+self.eps)*h + m)

		nodes = graph.nodes._replace(h=h)
		return graph._replace(nodes=nodes)


#=================================================================================================================
#=================================================================================================================
#=================================================================================================================


vmap2 = lambda f: jax.vmap(jax.vmap(f))

class AttentiveMessage(GraphModule):
	
	"""
	Used in: https://arxiv.org/abs/1910.00760
	"""
	#-------------------------------------------------------------------
	msg_mlp: nn.MLP
	att_mlp: nn.MLP
	adj_mask: bool
	#-------------------------------------------------------------------

	def __init__(
		self, 
		h_feats: int,
		msg_feats: int,
		*, 
		key: jax.Array,
		msg_mlp_width: int=16, 
		msg_mlp_depth: int=2,
		att_mlp_width: int=16, 
		att_mlp_depth: int=2,
		use_adj_mask: bool=True):
		
		key_msg, key_att = jr.split(key) 
		self.msg_mlp = nn.MLP(h_feats, msg_feats, msg_mlp_width, msg_mlp_depth, key=key_msg)
		self.att_mlp = nn.MLP(h_feats, 1, att_mlp_width, att_mlp_depth, final_activation=jnn.sigmoid, key=key_att)
		self.adj_mask = use_adj_mask

	#-------------------------------------------------------------------

	def apply_adj(self, graph: Graph, key: jax.Array) -> Graph:

		h = graph.nodes.h 
		x = h[:,None, :] - h[None] # NxNxH
		m = vmap2(self.msg_mlp)(x)
		a = vmap2(self.att_mlp)(x)
		if self.adj_mask:
			a = a*graph.edges.A
		m = jnp.sum(a*m, axis=1)
		nodes = graph.nodes._replace(h=m)
		return graph._replace(nodes=nodes)


#=================================================================================================================
#=================================================================================================================
#=================================================================================================================

def get_attention_weights(Q: jax.Array, K: jax.Array, mask: Optional[jax.Array]=None):
	w = jnp.dot(Q, K.T) / jnp.sqrt(Q.shape[-1])
	return jnn.softmax(w, where=mask, axis=1, initial=0) # N x N

def Attention(Q: Float[Array, "N dk"], K: Float[Array, "N dk"], 
			  V: Float[Array, "N dv"], mask: Optional[jax.Array]=None)->Float[Array, "N dv"]:
	w = get_attention_weights(Q, K, mask)
	return jnp.dot(w, V)

def MultiHeadAttention(Q: Float[Array, "N H dk"], K: Float[Array, "N H dk"], V: Float[Array, "N H dv"], mask: Optional[jax.Array]=None)->Float[Array, "N H dv"]:
	out = jax.vmap(Attention, in_axes=(1, 1, 1, None), out_axes=1)(Q, K, V, mask)
	return out


class GraphTransformer(GraphModule):
	
	"""
	paper: https://arxiv.org/pdf/2012.09699v2.pdf
	"""
	#-------------------------------------------------------------------
	# params :
	WQ: nn.Linear
	WK: nn.Linear
	WV: nn.Linear
	WO: nn.Linear
	W12: nn.MLP
	lnorm1: nn.LayerNorm
	lnorm2: nn.LayerNorm
	# statics : 
	n_heads: int
	use_mask: bool
	#-------------------------------------------------------------------

	def __init__(self, dk: int, dv: int, n_heads: int, use_mask: bool, *, key: jax.Array):
		"""
		dx (int): node features
		de (int): edge_features
		n_heads (int): number of attention heads 
		"""

		key_q, key_v, key_k, key_o, key_12 = jr.split(key, 5)
		
		self.WQ = nn.Linear(dk, dk*n_heads, key=key_q)
		self.WK = nn.Linear(dk, dk*n_heads, key=key_k)
		self.WV = nn.Linear(dk, dv*n_heads, key=key_v)
		self.WO = nn.Linear(dv*n_heads, dk, key=key_o)
		self.W12 = nn.MLP(dk, dk, 32, 1, use_bias=False, use_final_bias=False, key=key_12)

		self.n_heads = n_heads

		self.lnorm1 = nn.LayerNorm((dk,))
		self.lnorm2 = nn.LayerNorm((dk,))
		self.use_mask = use_mask

	#-------------------------------------------------------------------

	def apply_adj(self, graph: Graph, key: jax.Array)->Graph:
		"""return features aggregated through attention"""
		assert graph.edges.A is not None
		N = graph.nodes.h.shape[0]

		Q = jax.vmap(self.WQ)(graph.nodes.h).reshape((N, self.n_heads, -1))		# N x H x d
		K = jax.vmap(self.WK)(graph.nodes.h).reshape((N, self.n_heads, -1))		# N x H x d
		V = jax.vmap(self.WV)(graph.nodes.h).reshape((N, self.n_heads, -1))	    # N x H x dv

		mask = graph.edges.A if self.use_mask else None
		h = MultiHeadAttention(Q, K, V, mask).reshape((N, -1))		# N x H.dv
		h = jax.vmap(self.WO)(h)

		h = jax.vmap(self.lnorm1)(h+graph.h)
		h1 = jax.vmap(self.W12)(h)
		h2 = jax.vmap(self.lnorm2)(h+h1)

		nodes = graph.nodes._replace(h=h2)
		return graph._replace(nodes=nodes)


	#-------------------------------------------------------------------

class GAT(GraphModule):
	
	"""
	from: Graph Attention Networks
	paper: https://arxiv.org/pdf/1710.10903v3.pdf
	"""
	#-------------------------------------------------------------------
	att_fn: nn.MLP
	Wpre: nn.Linear
	Wpost: nn.Linear
	n_heads: int
	use_edges: bool
	adjacency_mask: bool
	sum_heads: bool
	#-------------------------------------------------------------------

	def __init__(self, in_features: int, out_features: int, n_heads: int, *, key: jax.Array,
				 att_depth: int=0, att_width: int=0, use_edges: bool=False, edge_features: int=0,
				 adjacency_mask: bool=True, sum_heads: bool=True):
		
		key_Wpre, key_att, key_Wpost = jr.split(key, 3)
		self.Wpre = nn.Linear(in_features, out_features, use_bias=False, key=key_Wpre)
		if not use_edges:
				self.att_fn = nn.MLP(2*out_features, n_heads, att_depth, att_width, key=key_att,
								 final_activation=linen.tanh, activation=linen.tanh)
		else:
			self.att_fn = nn.MLP(2*out_features+edge_features, n_heads, att_depth, att_width, key=key_att, 
								 final_activation=linen.tanh, activation=jnn.leaky_relu)
		self.n_heads = n_heads
		self.use_edges = use_edges
		self.adjacency_mask = adjacency_mask
		self.sum_heads = sum_heads

		self.Wpost = nn.Linear(in_features, out_features * n_heads, use_bias=False, key=key_Wpost)


	#-------------------------------------------------------------------

	def apply_adj(self, graph: Graph, key: jax.Array)->Graph:
		assert graph.edges.A is not None
		
		h_or = graph.nodes.h_learned
		N = h_or.shape[0]
		Wh = jax.vmap(self.Wpre)(h_or)
		if not self.use_edges:
			cat_Wh = jnp.concatenate(
				[jnp.repeat(Wh[:,None], N, axis=1),
				 jnp.repeat(Wh[None,:], N, axis=0)],
				 axis=-1
			) # N x N x 2F
		else:
			assert graph.edges.e is not None
			cat_Wh = jnp.concatenate(
				[jnp.repeat(Wh[:,None], N, axis=1),
				 jnp.repeat(Wh[None,:], N, axis=0),
				 graph.edges.e],
				 axis=-1
			) # N x N x 2F+E
		# Compute attention scores
		a = jax.vmap(jax.vmap(self.att_fn))(cat_Wh) # N x N x H
		# Get attention weights through softmax
		where = graph.edges.A[...,None] if self.adjacency_mask else None
		a = jnn.softmax(a, where=where, axis=0, initial=0.) # N x N x H
		# Aggregate transformed features accrding to attention weights
		h = jax.vmap(self.Wpost)(h_or).reshape((N, -1, self.n_heads)) # N x F x H

		if self.sum_heads:
			h = jax.vmap(aggregate, in_axes=-1, out_axes=-1)(h, a).sum(-1)
		else:
			h = jax.vmap(aggregate, in_axes=-1, out_axes=-1)(h, a).reshape((N, -1)) # N x F.H

		h = jnp.where(graph.nodes.m[:, None], h,h_or)
		nodes = graph.nodes._replace(h_learned=h)
		return graph._replace(nodes=nodes)



class RecurrentAttentionNetwork(GraphModule):
	
	"""
	from: Efficient Graph Generation with Graph Recurrent Attention Networks
	paper: https://proceedings.neurips.cc/paper/2019/file/d0921d442ee91b896ad95059d13df618-Paper.pdf
	"""
	#-------------------------------------------------------------------
	# Parameters:
	att_mp: AttentiveMessage
	rnn: PyTree
	# Statics:
	#-------------------------------------------------------------------

	def __init__(self, node_features: int, msg_features: int, *, key: jax.Array, 
				 att_mp_kws: dict={}, rnn_kws: dict={}, rnn_cell: Type=nn.GRUCell):
		
		key_att, key_rnn = jr.split(key)
		self.att_mp = AttentiveMessage(node_features, msg_features, key=key_att, **att_mp_kws)
		self.rnn = rnn_cell(msg_features, node_features, key=key_rnn, **rnn_kws)

	#-------------------------------------------------------------------

	def apply_adj(self, graph: Graph, key: jax.Array) -> Graph:

		h = graph.h
		m = self.att_mp(graph, key).h
		h = jax.vmap(self.rnn)(m, h)
		if graph.nodes.m is not None:
			h = h * graph.nodes.m[:,None]
		return eqx.tree_at(lambda G: G.nodes.h, graph, h)


#=================================================================================================================
#=================================================================================================================
#=================================================================================================================

class KNNAttentionConnector(GraphModule):
	
	"""
	"""
	#-------------------------------------------------------------------
	k: int
	query_fn: Callable
	key_fn: Callable
	#-------------------------------------------------------------------

	def __init__(self, k: int, state_dims: int, QK_dims: int, *, key: jax.Array,
				 query_fn: Optional[Any]=None, key_fn: Optional[Any]=None):
		
		self.k = k
		key_q, key_k = jr.split(key)
		self.query_fn = nn.Linear(state_dims, QK_dims, use_bias=False, key=key_q) if query_fn is None else query_fn
		self.key_fn = nn.Linear(state_dims, QK_dims, use_bias=False, key=key_k) if key_fn is None else key_fn

	#-------------------------------------------------------------------

	def __call__(self, graph: Graph, key: jax.Array)->Graph:
		
		Q = jax.vmap(self.query_fn)(graph.nodes.h)
		K = jax.vmap(self.key_fn)(graph.nodes.h)

		scores = jnn.softmax(Q @ K.T, axis=-1)
		values, _ = jax.lax.top_k(scores, self.k)
		min_values = jnp.min(values, axis=-1)
		A = (scores<min_values).astype(float)
		
		edges = graph.edges._replace(A=A)
		return graph._replace(edges = edges)

#=================================================================================================================
#=================================================================================================================
#=================================================================================================================


class RadiusConnector(GraphModule):
	
	"""
	Create connections between each node and every other node whose distance if < r
	"""
	#-------------------------------------------------------------------
	r: float
	position_getter: Callable[[Node], Float[Array, "D"]]
	#-------------------------------------------------------------------

	def __init__(self, r: float, position_getter: Callable = lambda n: n.p):
		
		self.r = r
		self.position_getter = position_getter

	#-------------------------------------------------------------------

	def apply_adj(self, graph: Graph, key: jax.Array)->Graph:
		assert graph.edges.A is not None

		p = self.position_getter(graph.nodes)
		mask = graph.nodes.m
		max_nodes = p.shape[0]
		mask = jnp.ones((max_nodes,)) if mask is None else mask

		dp = p[:, None, :] - p
		d = (dp*dp).sum(-1)

		A = (d < self.r).astype(float)

		edges = graph.edges._replace(A=A)

		return graph._replace(edges=edges)

#=================================================================================================================
#=================================================================================================================
#=================================================================================================================

class KNNConnector(GraphModule):

	"""
	Set edges of a graph as KNN conncetions
	"""
	#-------------------------------------------------------------------
	k: int
	position_getter: Callable[[Node], Float[Array, "D"]]
	#-------------------------------------------------------------------

	def __init__(self, k:int, position_getter: Callable = lambda n: n.p):
		
		self.k = k
		self.position_getter = position_getter

	#-------------------------------------------------------------------

	def apply_list(self, graph: Graph, *args, **kws)->Graph:
		assert graph.edges.senders is not None and graph.edges.receivers is not None
		
		p = self.position_getter(graph.nodes)
		mask = graph.nodes.m
		max_nodes = p.shape[0]
		mask = jnp.ones((max_nodes,)) if mask is None else mask

		dp = p[:, None, :] - p
		d = (dp*dp).sum(-1)
		d = jnp.where(mask[None, :], d, jnp.inf)
		_, idxs = jax.lax.top_k(-d, self.k)

		s = jnp.where(mask[:, None], idxs, max_nodes-1)
		r = jnp.where(mask[:, None], jnp.mgrid[:max_nodes, :self.k][0], max_nodes-1)

		s = s.reshape((-1,))
		r = r.reshape((-1,))

		edges = graph.edges._replace(senders=s, receivers=r)

		return graph._replace(edges=edges)

	#-------------------------------------------------------------------

	def apply_adj(self, graph: Graph, key: jax.Array)->Graph:
		assert graph.edges.A is not None

		p = self.position_getter(graph.nodes)
		mask = graph.nodes.m
		max_nodes = p.shape[0]
		mask = jnp.ones((max_nodes,)) if mask is None else mask

		dp = p[:, None, :] - p
		d = (dp*dp).sum(-1)
		values, _ = jax.lax.top_k(-d, self.k)
		minvals = values.min(-1)

		A = (-d < minvals).astype(float)

		edges = graph.edges._replace(A=A)

		return graph._replace(edges=edges)

#=================================================================================================================
#=================================================================================================================
#=================================================================================================================

class FullConnector(GraphModule):

	#-------------------------------------------------------------------
	#-------------------------------------------------------------------

	def apply_adj(self, graph: Graph, key: jax.Array) -> Graph:
		assert graph.edges.A is not None
		A = jnp.ones_like(graph.edges.A)
		edges = graph.edges._replace(A=A)
		return graph._replace(edges=edges)

	#-------------------------------------------------------------------

	def apply_list(self, graph: Graph, key: jax.Array) -> Graph:
		n = graph.nodes.h.shape[0]
		xy = jnp.mgrid[:n, :n].transpose((1,2,0)).reshape((-1, 2))
		r, s = xy.T
		edges = graph.edges._replace(receivers=r, senders=s)
		return graph._replace(edges=edges)


#=================================================================================================================
#=================================================================================================================
#=================================================================================================================

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


#=================================================================================================================
#=================================================================================================================
#=================================================================================================================


if __name__ == '__main__':
	f = CycleCountInjection()
	A = (jr.uniform(jr.PRNGKey(31), (5, 5)) < .3).astype(int)
	kc = eqx.filter_jit(f.count_cycles)(A)
	print(kc)






















