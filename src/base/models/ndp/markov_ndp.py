
from ast import Call
from typing import Any, Callable, Optional, TypeAlias, Union, Tuple
from jax._src.interpreters.mlir import eval_dynamic_shape_as_ivals
from jaxtyping import Float, PyTree

from src.models.base import DevelopmentalModel
from src.gnn.base import Graph, Node, Edge

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn

scaled_dot_product = lambda q, k: jnp.dot(q,k.T) / jnp.sqrt(k.shape[-1])



#==============================================================================
#===========================FEATURES EXTRACTOR=================================
#==============================================================================

from src.gnn.graph_features import (in_degrees, 
									out_degrees, 
									adjacency_powers,
									communicability)

def simple_rnn(h, w):
	h = jnn.tanh(jnp.dot(h,w))
	return h

class DynamicalFeaturesExtractor(eqx.Module):
	
	"""
	Compute the dynamical features (activations) of nodes i of a graph given
	a dynamical model and a generative model for inputs
	"""
	#-------------------------------------------------------------------
	dynamical_model: Callable
	generative_model: Callable
	init_fn: Callable
	iterations: int
	weight_extractor: Callable[[Graph], jax.Array]

	#-------------------------------------------------------------------

	@property
	def n(self)->int:
		return self.iterations

	#-------------------------------------------------------------------

	def __init__(
		self, 
		dynamical_model: Callable,
		init_fn: Callable,
		generative_model: Callable,
		iterations: int,
		weight_extractor: Callable= lambda G: G.edges.e):
		
		self.dynamical_model = dynamical_model
		self.generative_model = generative_model
		self.init_fn = init_fn
		self.iterations = iterations
		self.weight_extractor = weight_extractor

	#-------------------------------------------------------------------

	def __call__(self, graph: Graph, key: jax.Array)->jax.Array:
		
		key_init, key_x = jr.split(key)
		w = self.weight_extractor(graph)
		def step(h, x):
			h_ = self.dynamical_model(h, w)
			return h_, h
		x = self.generative_model(key_x)
		h0 = self.init_fn(key_init).at[:x.shape[0]].set(x)
		_, hs = jax.lax.scan(step, h0, None, self.iterations)
		return hs.T


class EdgeFeaturesExtractor(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	adj_powers: bool=True
	powmax: int=3
	communicability: bool=False
	#-------------------------------------------------------------------

	@property
	def n(self)->int:
		return (4
				+ int(self.adj_powers) * self.powmax
				+ int(self.communicability))

	#-------------------------------------------------------------------

	def __call__(self, graph: Graph, key: Optional[jax.Array]=None)->jax.Array:
		
		features = [graph.A[...,None], graph.A.T[...,None], jnp.identity(graph.N)[...,None]]
		if self.adj_powers:
			pows = adjacency_powers(graph.A, self.powmax)# p x N x N
			pows = pows.transpose((1,2,0))
			features.append(pows) 

		if self.communicability:
			comm = communicability(graph.A)[...,None]
			features.append(comm)
		features = jnp.concatenate(features, axis=-1)
		return features


class StructuralFeaturesExtractor(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	degree: bool
	neighbor_degree: bool
	#-------------------------------------------------------------------

	def __init__(self,
				 degree: bool=True,
				 neighbor_degree: bool=True):
		self.degree = degree
		self.neighbor_degree = neighbor_degree

	#-------------------------------------------------------------------

	@property
	def n(self)->int:
		return (int(self.degree) * 3 
				+ int(self.neighbor_degree)*4)

	#-------------------------------------------------------------------

	def __call__(self, graph: Graph, key: Optional[jax.Array]=None)->jax.Array:

		assert graph.edges.A is not None
		A = graph.edges.A
		in_degree = in_degrees(A)
		out_degree = out_degrees(A)
		degree = in_degree + out_degree

		avg_in_neighbor_in_degree = jnp.dot(A.T, in_degree)
		avg_in_neighbor_out_degree = jnp.dot(A.T, out_degree)
		avg_out_neighbor_in_degree = jnp.dot(A, in_degree)
		avg_out_neighbor_out_degree = jnp.dot(A, out_degree)

		features = []
		if self.degree:
			features = features + [degree, in_degree, out_degree]
		if self.neighbor_degree:
			features = features + [avg_in_neighbor_in_degree, 
								   avg_in_neighbor_out_degree,
								   avg_out_neighbor_in_degree,
								   avg_out_neighbor_out_degree]

		return jnp.concatenate([f[:,None] for f in features], axis=-1)

class FeatureExtractor(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	structural_feature_extractor: StructuralFeaturesExtractor
	dynamical_feature_extractor: Optional[DynamicalFeaturesExtractor]
	edge_feature_extractor: Optional[EdgeFeaturesExtractor]
	add_edge_features: bool
	input_dims: int
	action_dims: int
	node_memory_features: Optional[int]
	node_memory_fn: Optional[Callable]
	#-------------------------------------------------------------------

	def __init__(self, 
				 n_nodes: int, 
				 input_dims: int, 
				 action_dims: int,
				 dynamical_iterations: int=0,
				 generative_model: Optional[Callable]=None, 
				 add_edge_features: bool=False,
				 node_memory_features: Optional[int]=None,
				 node_features: Optional[int]=None):

		self.add_edge_features = add_edge_features
		self.structural_feature_extractor = StructuralFeaturesExtractor()
		
		if dynamical_iterations:
			if generative_model is None:
				generative_model = lambda key: jr.normal(key, (input_dims,))
			self.dynamical_feature_extractor = DynamicalFeaturesExtractor(
				dynamical_model = simple_rnn,
				init_fn = lambda _: jnp.zeros((n_nodes, )),
				generative_model = generative_model,
				iterations = dynamical_iterations
			)
		else :
			self.dynamical_feature_extractor = None
		
		if add_edge_features:
			self.edge_feature_extractor = EdgeFeaturesExtractor()
		else:
			self.edge_feature_extractor = None
		self.input_dims = input_dims
		self.action_dims = action_dims

		self.node_memory_features = node_memory_features
		if node_memory_features is None:
			self.node_memory_fn = None
		else:
			assert node_features is not None
			self.node_memory_fn = nn.Linear(node_features, node_memory_features, key=jr.key(1))

	#-------------------------------------------------------------------
	@property
	def n(self):
		return self.node_features
	@property
	def node_features(self):
		dn = self.dynamical_feature_extractor.n if self.dynamical_feature_extractor is not None else 0
		return dn + self.structural_feature_extractor.n + 2
	@property
	def edge_features(self):
		return (self.edge_feature_extractor.n 
				if   self.edge_feature_extractor is not None 
				else 0)
	#-------------------------------------------------------------------

	def __call__(self, graph: Graph, key: jax.Array)->Graph:

		assert graph.edges.A is not None
		assert graph.nodes.m is not None
		structural_features = self.structural_feature_extractor(graph)
		input_ids = jnp.zeros((graph.N,)).at[:self.input_dims].set(jnp.arange(self.input_dims)+1.)[:,None]
		N = graph.nodes.m.sum().astype(int)
		output_ids =  jax.lax.dynamic_update_slice(jnp.zeros((graph.N,)), jnp.arange(self.action_dims).astype(float)+1., (N-self.action_dims,))
		features = [input_ids, output_ids[...,None], structural_features]
		if self.dynamical_feature_extractor is not None:
			dynamical_features = self.dynamical_feature_extractor(graph, key)
			features.append(dynamical_features)
		node_features = jnp.concatenate(features, axis=-1)
		if self.edge_feature_extractor is not None:
			edge_features = [graph.A]	
			edge_features = self.edge_feature_extractor(graph)
		else:
			edge_features = graph.edges.e

		return eqx.tree_at(lambda G: [G.nodes.h, G.edges.e], graph, [node_features, edge_features])


class MarkovNDP(DevelopmentalModel):
	
	"""
	"""
	#-------------------------------------------------------------------

	#-------------------------------------------------------------------

	def __call__(self, graph: Graph, key: jax.Array)->Graph:
		
		key_feat, key_gnn, key_morph = jr.split(key, 3)
		graph = self.get_init_features(graph, key_feat)
		graph = self.gnn(graph, key_gnn)
		graph = self.transform_graph(graph, key_morph)

		return graph

	#-------------------------------------------------------------------

	def get_init_features(self, graph: Graph, key: jax.Array)->Graph:

		raise NotImplementedError

	#-------------------------------------------------------------------

	def gnn(self, graph: Graph, key: jax.Array)->Graph:
		
		raise NotImplementedError

	#-------------------------------------------------------------------

	def transform_graph(self, graph: Graph, key: jax.Array)->Graph:

		raise NotImplementedError

	#-------------------------------------------------------------------


class QueryKeyMarkovNDP(MarkovNDP):
	
	"""
	"""
	#-------------------------------------------------------------------
	QK_fn: nn.Linear
	node_policy: nn.Linear
	global_policy: nn.Linear
	theta_add: Float
	theta_prune: Float
	score_activation: Callable
	policy_activation: Callable
	weighted: bool
	weight_activation: Callable
	weight_features: int
	synaptic_pruning: bool
	#-------------------------------------------------------------------

	def __init__(self, node_features: int, qk_features: int, global_features: int, theta_add: float, 
				 theta_prune: float, *, key: jax.Array, learnable_thresholds: bool=False,
				 score_activation: Callable=jnn.sigmoid, policy_activation: Callable=jnn.sigmoid, weighted: bool=False,
				 weight_activation: Callable=jnn.tanh, weight_features: int=1, synaptic_pruning: bool=True):
		
		out_heads = 2 + int(weighted)
		QK_key, node_policy_key, global_policy_key = jr.split(key, 3)
		self.QK_fn = nn.Linear(node_features, qk_features*2*out_heads, key=QK_key)
		self.node_policy = nn.Linear(node_features, 2, key=node_policy_key)
		self.global_policy = nn.Linear(global_features, 2, key=global_policy_key)
		if learnable_thresholds:
			self.theta_add = jnp.array([theta_add])
			self.theta_prune = jnp.array([theta_prune])
		else:
			self.theta_add = theta_add
			self.theta_prune = theta_prune
		self.score_activation = score_activation
		self.weighted = weighted
		self.weight_activation = weight_activation
		self.weight_features = weight_features
		self.synaptic_pruning = synaptic_pruning
		self.policy_activation = policy_activation

	#-------------------------------------------------------------------

	def transform_graph(self, graph: Graph, key: jax.Array)->Graph:

		assert graph.edges.A is not None
		assert graph.global_ is not None
		assert graph.nodes.m is not None

		scaled_dot_product = lambda q, k: jnp.dot(q,k.T) / jnp.sqrt(k.shape[-1])

		node_actions = self.policy_activation(jax.vmap(self.node_policy)(graph.h))
		global_gate = self.policy_activation(self.global_policy(graph.global_))
		node_actions = node_actions * global_gate[None,:]

		key_grow, key_sg = jr.split(key)
		alive_mask = graph.nodes.m.astype(bool)
		grow = (jr.uniform(key_grow, (graph.N,)) < node_actions[:, 0]) & alive_mask # Add nodes
		sg = (jr.uniform(key_sg, (graph.N,)) < node_actions[:, 1]) & alive_mask   # Add edges

		# Add new nodes
		n_grow = grow.sum()
		mask = graph.nodes.m
		N = mask.shape[0]
		n_alive = mask.sum()
		A = graph.edges.A
		new_n = n_alive+n_grow
		new_mask = (jnp.arange(graph.nodes.h.shape[0]) < new_n).astype(float)
		xnew_mask = new_mask - mask
		# compute childs index for each parent: pc[parent] = child index
		pc = (jnp.where(grow, jnp.cumsum(grow)-1, -1) + (n_alive*grow)).astype(int)
		# Set child´s incoming connections (parent)
		nA = jax.ops.segment_sum(jnp.clip(jnp.identity(N), 0., 1.).T, pc, N).T
		A = jnp.where(xnew_mask[None,:], nA, A) * new_mask[None,:] * new_mask[:,None]

		qk = jax.vmap(self.QK_fn)(graph.h)
		if self.weighted:
			qk_add, qk_prune, qk_weight = jnp.split(qk, 3, axis=-1)
		else:
			qk_add, qk_prune = jnp.split(qk, 2, axis=-1)

		# Add edges
		q_add, k_add = jnp.split(qk_add, 2, axis=-1)
		sdp_add = self.score_activation(scaled_dot_product(q_add, k_add))
		sdp_add = sdp_add * graph.nodes.m[:,None] * graph.nodes.m[None, :] * sg.astype(float)[:,None]
		added_edges = (sdp_add > jnp.clip(self.theta_add, .01, jnp.inf)).astype(float) 
		A = jnp.clip(A+added_edges, 0., 1.)

		# Remove edges
		if self.synaptic_pruning:
			q_prune, k_prune = jnp.split(qk_prune, 2, axis=-1)
			sdp_prune = self.score_activation(scaled_dot_product(q_prune, k_prune))
			sdp_prune = sdp_prune * graph.nodes.m[:,None] * graph.nodes.m[None, :] * sg.astype(float)[:,None]
			pruned_edges = (sdp_prune > self.theta_prune).astype(float)
			A = jnp.clip(A - pruned_edges, 0., 1.)

		if self.weighted:
			q_weight, k_weight = jnp.split(qk_weight, 2, axis=-1) #type: ignore
			w = self.weight_activation(scaled_dot_product(q_weight, k_weight))
			w = w * A
			return eqx.tree_at(lambda G: [G.nodes.m, G.edges.A, G.edges.e], graph, [new_mask, A, w])

		else:
			return eqx.tree_at(lambda G: [G.nodes.m, G.edges.A], graph, [new_mask, A])




class MNDP(MarkovNDP):
	
	"""
	"""
	#-------------------------------------------------------------------
	feature_extractor_: FeatureExtractor
	gnn_: Callable
	node_policy: Callable
	global_policy: Callable
	target_fn: Callable
	prune_fn: Optional[Callable]
	weight_fn: Optional[Callable]
	edge_features: Optional[int]
	stochastic_actions: bool
	stochastic_sg: bool
	theta_add: float
	theta_prune: float
	max_nodes: int
	init_nodes: int
	global_features: int
	#-------------------------------------------------------------------

	def __init__(self, 
				 gnn: Callable,
				 feature_extractor: FeatureExtractor,
				 node_features: int,
				 global_features: int,
				 init_nodes: int,
				 max_nodes: int=128,
				 edge_features: Optional[int]=None,
				 synaptic_pruning: bool=False,
				 weighted: bool=True,
				 weight_activation: Callable=lambda x: x,
				 stochastic_actions: bool=True,
				 stochastic_sg: bool=False,
				 *,
				 key: jax.Array):
		
		self.gnn_ = gnn
		self.feature_extractor_ = feature_extractor
		node_policy_key, global_policy_key, tgt_key, prune_key, w_key = jr.split(key, 5)
		self.node_policy = nn.MLP(node_features, 2, 16, 1, key=node_policy_key, final_activation=jnn.sigmoid)
		self.global_policy = nn.MLP(global_features, 2, 16, 1, key=global_policy_key, final_activation=jnn.sigmoid)
		self.target_fn = nn.Linear(node_features, 8, key=tgt_key)
		if synaptic_pruning:
			self.prune_fn = nn.Linear(node_features, 1, key=prune_key)
		else: 
			self.prune_fn = None
		if weighted:
			ef = 1 if edge_features is None else edge_features
			self.weight_fn = nn.MLP(node_features*2, ef, 16, 1, key=w_key, final_activation=weight_activation)
		else:
			self.weight_fn = None
		self.theta_add = 1.
		self.theta_prune = 1.
		self.edge_features = edge_features
		self.global_features = global_features
		self.max_nodes = max_nodes
		self.init_nodes = init_nodes
		self.stochastic_sg = stochastic_sg
		self.stochastic_actions = stochastic_actions
	#-------------------------------------------------------------------

	def weight(self, graph: Graph, key: jax.Array)->Graph:
		if self.weight_fn is not None:
			cat_h = jnp.concatenate(
				[jnp.repeat(graph.h[:,None], graph.N, axis=1),
				 jnp.repeat(graph.h[None,:], graph.N, axis=0)],
				 axis=-1
			)
			w = jax.vmap(jax.vmap(self.weight_fn))(cat_h)
			return graph.replace(e=w*graph.A[...,None])
		return graph

	def rollout(self, state: Graph, key: jr.PRNGKeyArray, steps: int) -> Tuple[Graph, Graph]:
		key_roll, key_w = jr.split(key)
		graph, graphs = super().rollout(state, key_roll, steps)
		graph = self.weight(graph, key_w)
		return graph, graphs

	def rollout_(self, state: Graph, key: jr.PRNGKeyArray, steps: int) -> Graph:
		key_roll, key_w = jr.split(key)
		graph = super().rollout_(state, key_roll, steps)
		graph = self.weight(graph, key_w)
		return graph
	#-------------------------------------------------------------------

	def transform_graph(self, graph: Graph, key: jax.Array)->Graph:

		assert graph.edges.A is not None
		assert graph.global_ is not None
		assert graph.nodes.m is not None

		node_actions = jax.vmap(self.node_policy)(graph.h)
		global_gate = self.global_policy(graph.global_)
		node_actions = node_actions * global_gate[None,:]

		key_grow, key_sg, key_w = jr.split(key, 3)
		alive_mask = graph.nodes.m.astype(bool)
		if self.stochastic_actions:
			grow = (jr.uniform(key_grow, (graph.N,)) < node_actions[:, 0]) & alive_mask # Add nodes
			sg = (jr.uniform(key_sg, (graph.N,)) < node_actions[:, 1]) & alive_mask   # Add edges
		else:
			grow = (node_actions[:, 0] > 0.5) & alive_mask
			sg = (node_actions[:, 1] > 0.5) & alive_mask

		# Add new nodes
		n_grow = grow.sum()
		mask = graph.nodes.m
		N = mask.shape[0]
		n_alive = mask.sum()
		A = graph.edges.A
		new_n = n_alive+n_grow
		new_mask = (jnp.arange(graph.nodes.h.shape[0]) < new_n).astype(float)
		xnew_mask = new_mask - mask
		# compute childs index for each parent: pc[parent] = child index
		pc = (jnp.where(grow, jnp.cumsum(grow)-1, -1) + (n_alive*grow)).astype(int)
		# Set child´s incoming connections (parent)
		nA = jax.ops.segment_sum(jnp.clip(jnp.identity(N), 0., 1.).T, pc, N).T
		A = jnp.where(xnew_mask[None,:], nA, A) * new_mask[None,:] * new_mask[:,None]

		# Add edges
		q_add, k_add = jnp.split(jax.vmap(self.target_fn)(graph.h), 2, axis=-1)
		sdp_add = scaled_dot_product(q_add, k_add)
		sdp_add = sdp_add * graph.nodes.m[:,None] * graph.nodes.m[None, :] * sg.astype(float)[:,None]
		if not self.stochastic_sg:
			added_edges = (sdp_add > self.theta_add).astype(float) 
		else:
			_, key = jr.split(key_sg)
			added_edges = (jr.uniform(key, sdp_add.shape)<jnn.sigmoid(sdp_add)).astype(float)
		A = jnp.clip(A+added_edges, 0., 1.)

		# Remove edges
		if self.prune_fn is not None:
			q_prune, k_prune = jnp.split(jax.vmap(self.prune_fn)(graph.h), 2)
			prune_score = scaled_dot_product(q_prune, k_prune)
			pruned_edges = (prune_score > self.theta_prune).astype(float)
			A = jnp.clip(A - pruned_edges, 0., 1.)

		graph = eqx.tree_at(lambda G: [G.nodes.m, G.edges.A], graph, [new_mask, A])

		return graph

	#-------------------------------------------------------------------

	def gnn(self, graph: Graph, key: jax.Array) -> Graph:
		G = self.gnn_(graph, key)
		return G

	#-------------------------------------------------------------------

	def get_init_features(self, graph: Graph, key: jax.Array) -> Graph:
		graph = self.feature_extractor_(graph, key)
		return graph

	#-------------------------------------------------------------------

	def initialize(self, key: jr.PRNGKeyArray) -> Graph:
		V = Node(h = jnp.zeros((self.max_nodes, self.feature_extractor_.node_features)), 
				 m=jnp.zeros((self.max_nodes,)).at[:self.init_nodes].set(1.))
		E = Edge(A=jnp.zeros((self.max_nodes,self.max_nodes)).astype(float), 
				 e=jnp.zeros((self.max_nodes,self.max_nodes, self.edge_features)))
		global_ = jnp.zeros((self.global_features, ))
		G = Graph(nodes=V, edges=E, global_=global_)
		return G
