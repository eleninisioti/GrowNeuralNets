import jax
from jax._src.lax.fft import _naive_rfft
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn
from typing import Mapping, Tuple, Optional

from src.gnn.base import Graph
import src.gnn.graph_features as gf
import src.gnn.layers as gnn



class DenoisingNetwork(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	# Parameters:

	# Statics:
	
	#-------------------------------------------------------------------

	def __init__(self):
		pass

	#-------------------------------------------------------------------

	def __call__(self, graph: Graph, key: jax.Array)->Tuple[jax.Array, jax.Array]:
		"""
		Take as input a (noisy) graph and returns a graph where h (N, Cx) is the probability distribution over 
			node classes, and e (N, N, Ce) is the distribution over edge classes
		"""
		
		raise NotImplementedError

	#-------------------------------------------------------------------


class DummyDenoisingNetwork(DenoisingNetwork):

	#-------------------------------------------------------------------

	def __call__(self, graph: Graph, key: jax.Array) -> Tuple[jax.Array, jax.Array]:
		assert graph.edges.e is not None
		return (jnp.ones_like(graph.h)/graph.h.shape[-1], 
				jnp.ones((graph.edges.e.shape[0]**2, graph.edges.e.shape[-1]))/graph.edges.e.shape[-1])

	#-------------------------------------------------------------------




#=======================================================================
#-------------------------------DiGress Model---------------------------
#=======================================================================


class FiLM(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	# Parameters:
	W_add: nn.Linear
	W_mul: nn.Linear
	#-------------------------------------------------------------------

	def __init__(self, in_features: int, out_features: int, key: jax.Array):
		
		key_add, key_mul = jr.split(key)
		self.W_add = nn.Linear(in_features, out_features, key=key_add)
		self.W_mul = nn.Linear(in_features, out_features, key=key_mul)

	#-------------------------------------------------------------------

	def __call__(self, M1: jax.Array, M2: jax.Array):
		
		if len(M1.shape) == 3:
			M1_mul = jax.vmap(jax.vmap(self.W_mul))(M1)
			M1_add = jax.vmap(jax.vmap(self.W_add))(M1) 
		else:
			M1_mul = jax.vmap(self.W_mul)(M1)
			M1_add = jax.vmap(self.W_add)(M1)

		return M1_add + M1_mul * M2 + M2

class PNA(eqx.Module):
	#-------------------------------------------------------------------
	W: nn.Linear
	#-------------------------------------------------------------------
	def __init__(self, out_features: int, *, key: jax.Array):
		self.W = nn.Linear(4, out_features, key=key)
	#-------------------------------------------------------------------
	def __call__(self, x: jax.Array):
		x = jnp.concatenate([x.max(), x.min(), x.mean(), x.std()])
		return self.W(x)
	#-------------------------------------------------------------------

def scaled_dot_product(Q, K):
	return jnp.dot(Q, K.T) / jnp.sqrt(K.shape[-1])


class DigressTransformerLayer(gnn.GraphModule):
	
	"""
	"""
	#-------------------------------------------------------------------
	# Parameters:
	Q: nn.Linear
	K: nn.Linear
	V: nn.Linear
	film_E_w: FiLM
	film_y_E: Optional[FiLM]
	film_y_X: Optional[FiLM]
	E_out: nn.Linear
	X_out: nn.Linear
	# Statics:
	n_heads: int
	has_global_features: bool
	#-------------------------------------------------------------------

	def __init__(self, dx: int, de: int, n_heads: int, dqk: int, dv: int, 
				 *, key: jax.Array, has_global_features: bool=False, dy: Optional[int]=None):
		
		self.n_heads = n_heads

		key, kQ, kK, kV = jr.split(key, 4)
		
		self.Q = nn.Linear(dx, dqk*n_heads, key=kQ)
		self.K = nn.Linear(dx, dqk*n_heads, key=kK)
		self.V = nn.Linear(dx, dv*n_heads, key=kV)

		key, kEw, kyX, kyE = jr.split(key, 4)

		self.film_E_w = FiLM(de, n_heads*dqk, key=kEw)
		self.has_global_features = has_global_features
		if has_global_features:
			assert dy is not None
			self.film_y_E = FiLM(dy, dqk*n_heads, key=kyE)
			self.film_y_X = FiLM(dy, dv*n_heads, key=kyX)
		else:
			self.film_y_E = None
			self.film_y_X = None

		kX, kE = jr.split(key)

		self.E_out = nn.Linear(dqk*n_heads, de, key=kE)
		self.X_out = nn.Linear(dv*n_heads, dx, key=kX)

	#-------------------------------------------------------------------

	def __call__(self, graph: Graph, key: Optional[jax.Array]=None)->Graph:
		
		assert graph.edges.e is not None

		X, E, N = graph.h, graph.edges.e, graph.N

		q = jax.vmap(self.Q)(X) # (N, h*dqk)
		k = jax.vmap(self.K)(X) # (N, h*dqk)
		v = jax.vmap(self.V)(X).reshape((N, self.n_heads, -1)) # (N, h*dv)

		Y = q[:, None, ...] * k[None, ...]                                  # (N, N, h*dqk)
		YE = self.film_E_w(E, Y) # (N, N, h*dqk)
		YEh = YE.reshape((N, N, self.n_heads, -1))                          # (N, N, h, dqk)

		W = jnn.softmax(YEh.sum(-1), axis=1) # (N, N, h)

		M = jax.vmap(jnp.dot, in_axes=(-1, 1), out_axes=1)(W, v).reshape((N, -1)) # (N, h*dv)

		if self.has_global_features:
			assert graph.global_ is not None and self.film_y_E is not None and self.film_y_X is not None
			y = graph.global_ # (dy,)
			E = self.film_y_E(y[None, None,:], YE) # (N, N, h*dqk)
			X = self.film_y_X(y[None, :], M)       # (N, h*dv)

		else:
			E = YE # (N, N, h*dqk)
			X = M  # (N, h*dv)

		E = jax.vmap(jax.vmap(self.E_out))(E) # (N, N, de)
		X = jax.vmap(self.X_out)(X)			  # (N, dx)

		return eqx.tree_at(lambda G: [G.nodes.h, G.edges.e], graph, [X, E])


class FeaturesExtractor(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	degree_features: bool
	cycle_features: bool
	#-------------------------------------------------------------------

	def __init__(self, 
				 degree_features: bool=True, 
				 cycle_features: bool=False):
		
		self.degree_features = degree_features
		self.cycle_features = cycle_features

	#-------------------------------------------------------------------

	@property
	def n(self):
		return 7 * int(self.degree_features) + 4 * int(self.cycle_features)

	#-------------------------------------------------------------------

	def __call__(self, graph: Graph, key: Optional[jax.Array]=None):

		assert graph.E is not None

		A = graph.E[..., 1]
		in_degree = gf.in_degrees(A)
		out_degree = gf.out_degrees(A)
		degree = in_degree + out_degree

		avg_in_neighbor_in_degree = jnp.dot(A.T, in_degree)
		avg_in_neighbor_out_degree = jnp.dot(A.T, out_degree)
		avg_out_neighbor_in_degree = jnp.dot(A, in_degree)
		avg_out_neighbor_out_degree = jnp.dot(A, out_degree)

		features = []
		if self.degree_features:
			features = features + [degree[:,None], 
								   in_degree[:,None], 
								   out_degree[:,None],
								   avg_in_neighbor_in_degree[:,None], 
								   avg_in_neighbor_out_degree[:,None],
								   avg_out_neighbor_in_degree[:,None],
								   avg_out_neighbor_out_degree[:,None]]
		if self.cycle_features:
			ks = jnp.arange(6)[2:]
			kcycles = jax.vmap(gf.count_k_cycles, in_axes=(None, 0))(A, ks)
			features = features + [kcycles]

		return jnp.concatenate(features, axis=-1)


class GraphTransformerDenoisingNetwork(DenoisingNetwork):
	
	"""
	"""
	#-------------------------------------------------------------------
	# Parameters:
	gt_layers:     list[DigressTransformerLayer]
	in_X:  nn.MLP
	out_X: nn.MLP
	in_E:  nn.MLP
	out_E: nn.MLP
	# Statics:
	feature_extractor: FeaturesExtractor
	#-------------------------------------------------------------------

	def __init__(self, 
				 gt_layers: int,
				 dx: int,
				 de: int,
				 gt_x_features: int,
				 gt_e_features: int,
				 transformer_heads: int,
				 dqk: int,
				 dv: int, 
				 in_mlp_depth: int,
				 in_mlp_width: int,
				 out_mlp_depth: int,
				 out_mlp_width: int,
				 degree_features: bool,
				 cycle_features: bool,
				 has_global_features: bool=False,
				 *,
				 key: jax.Array):
		
		gt_keys, iX, oX, iE, oE = jr.split(key, 5)
		gt_keys = jr.split(gt_keys, gt_layers)

		layers = [
			DigressTransformerLayer(gt_x_features, gt_e_features, transformer_heads,
				dqk, dv, has_global_features=has_global_features,
				key=gt_keys[i])
		for i in range(gt_layers)]
		self.gt_layers = layers

		self.feature_extractor = FeaturesExtractor(degree_features=degree_features,
												   cycle_features=cycle_features)
		self.in_X = nn.MLP(dx+self.feature_extractor.n, gt_x_features, in_mlp_width, 
						   in_mlp_depth, key=iX)
		self.out_X = nn.MLP(gt_x_features, dx, out_mlp_width, out_mlp_depth, key=oX)
		self.in_E = nn.MLP(de, gt_e_features, in_mlp_width, in_mlp_depth, key=iE)
		self.out_E = nn.MLP(gt_e_features, de, out_mlp_width, out_mlp_depth, key=oE)

	#-------------------------------------------------------------------

	def __call__(self, graph: Graph, key: jax.Array)->Tuple[jax.Array, jax.Array]:
		""""""
		F = self.feature_extractor(graph, None)
		graph = eqx.tree_at(lambda G: G.nodes.h, 
							graph, 
							jnp.concatenate([graph.h, F], 
											 axis=-1))

		assert graph.edges.e is not None

		X = jax.vmap(self.in_X)(graph.h)
		E = jax.vmap(jax.vmap(self.in_E))(graph.edges.e)

		graph = eqx.tree_at(lambda G: [G.nodes.h, G.edges.e], graph, [X, E])

		for layer in self.gt_layers:
			graph = layer(graph)

		assert graph.edges.e is not None	

		X = jax.vmap(self.out_X)(graph.h)
		E = jax.vmap(jax.vmap(self.out_E))(graph.edges.e) 

		X = jnn.softmax(X, axis=-1)
		E = jnn.softmax(E, axis=-1)

		return X, E




