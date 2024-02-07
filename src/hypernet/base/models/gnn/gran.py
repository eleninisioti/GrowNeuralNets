from dataclasses import InitVar
from src.models.gnn.base import GraphModule, Graph, Node, Edge

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn


vmap2 = lambda f: jax.vmap(jax.vmap(f))

class AttentiveMessage(GraphModule):
	
	"""
	"""
	#-------------------------------------------------------------------
	msg_mlp: nn.MLP
	att_mlp: nn.MLP
	#-------------------------------------------------------------------

	def __init__(
		self, 
		h_feats: int,
		msg_feats: int,
		*, 
		key: jr.PRNGKeyArray,
		msg_mlp_width: int=16, 
		msg_mlp_depth: int=2):
		
		key_msg, key = jr.split(key) 
		self.msg_mlp = nn.MLP(h_feats, msg_feats, msg_mlp_width, msg_mlp_depth, key=key_msg)
		self.att_mlp = nn.MLP(h_feats+1, 1, msg_mlp_width, msg_mlp_depth, final_activation=jnn.sigmoid, key=key_msg)

	#-------------------------------------------------------------------

	def __call__(self, graph: Graph, mask: jax.Array, key: jr.PRNGKeyArray) -> Graph:

		h = graph.nodes.h 
		x = h[:,None, :] - h[None] # NxNxH
		m = vmap2(self.msg_mlp)(x)
		a = jax.vmap(self.att_mlp)(jnp.concatenate([x, mask[:,None]], axis=-1))
		m = jnp.sum(a*m, axis=1)

		nodes = graph.nodes._replace(h=m)
		return graph._replace(nodes=nodes)
		

class GRAN(GraphModule):
	
	"""
	Paper: Efficient Graph Generation with Graph Recurrent Attention Networks
	Link: https://arxiv.org/abs/1910.00760
	"""
	#-------------------------------------------------------------------
	R: int
	msg_fn: AttentiveMessage
	gru: nn.GRUCell
	init_fn = nn.Linear
	mlp_alpha: nn.MLP
	mlp_theta: nn.MLP
	#-------------------------------------------------------------------

	def __init__(
		self, 
		R: int,
		K: int,
		h_feats: int,
		msg_feats: int,
		*, 
		key: jr.PRNGKeyArray,
		msg_mlp_width: int=16, 
		msg_mlp_depth: int=2):
		
		self.R = R
		msg_key, gru_key, init_key, alpha_key, theta_key = jr.split(key, 5)
		self.msg_fn = AttentiveMessage(h_feats, msg_feats, key=msg_key, 
			msg_mlp_width=msg_mlp_width, msg_mlp_depth=msg_mlp_depth)
		self.gru = nn.GRUCell(msg_feats, h_feats, key=gru_key)
		self.init_fn = nn.Linear(h_feats, h_feats, key=init_key)
		self.mlp_alpha = nn.MLP(h_feats, K, 16, 2, key=alpha_key)
		self.mlp_theta = nn.MLP(h_feats, K, 16, 2, key=theta_key)

	#-------------------------------------------------------------------

	def apply_adj(self, graph: Graph, key: jr.PRNGKeyArray) -> Graph:
		
		def step(carry, i):
			
			graph, key = carry
			key, key_gnn = jr.split(key)
			graph = self.apply_gnn(graph, i, key_gnn)
			h = graph.nodes.h
			x = (h[i][None,...] - h).sum(0)
			alpha = jnn.softmax(jax.vmap(self.mlp_alpha)(x))
			theta = jnn.sigmoid(jax.vmap(self.mlp_theta))(x)

		return graph

	#-------------------------------------------------------------------

	def apply_gnn(self, graph: Graph, step: int, key: jr.PRNGKeyArray):

		max_nodes = graph.nodes.h.shape[0]
		growth_mask = jnp.zeros((max_nodes,)).at[step].set(1.)
		
		def gnn_step(carry, x):
			g, k = carry
			k, km = jr.split(k)
			m = self.msg_fn(g, growth_mask, km).nodes.h
			h = jax.vmap(self.gru)(m, g.nodes.h)
			nodes = g.nodes._replace(h=h)
			return [g._replace(nodes=nodes), k], None

		[graph, _], _ = jax.lax.scan(gnn_step, [graph, key], None, self.R)
		return graph





		
