import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn
from typing import Tuple
from jaxtyping import Int
from warnings import warn

from src.gnn.base import Graph, Node, Edge
from src.gnn.graph_features import degrees, in_degrees, out_degrees
from src.models.diffusion.noise import DiscreteNoiseModel, NoiseModel
from src.models.diffusion.denoising import DenoisingNetwork

def cross_entropy_loss(pX, pE, X, E, lmbda=1.):
	X_loss = jnp.sum(-jnp.sum(X * jnp.log(pX+1e-6), axis=-1))
	E_loss = jnp.sum(-jnp.sum(E * jnp.log(pE+1e-6), axis=-1))
	return X_loss + lmbda * E_loss

def EtoA(E: jax.Array):
	return E[...,1]

def make_graph(X: jax.Array, E: jax.Array):
	nodes = Node(h=X)
	edges = Edge(A=EtoA(E), e=E)
	return Graph(nodes=nodes, edges=edges)

class DiffusionModel(eqx.Module):
	"""
	"""
	#-------------------------------------------------------------------
	denoising_network: DenoisingNetwork
	noise_model: NoiseModel
	ordering: str
	#-------------------------------------------------------------------

	def __init__(self, 
				 noise_model: NoiseModel, 
				 denoising_network: DenoisingNetwork,
				 ordering: str="degree"):
		
		self.noise_model = noise_model
		self.denoising_network = denoising_network
		self.ordering = ordering

	#-------------------------------------------------------------------

	def __call__(self, graph: Graph, key: jax.Array) -> Tuple[jax.Array, jax.Array]:
		"""
		Inputs:
			graph (Graph): a noisy graph
			key (PRNGKey)
		Returns:
			pX: nodes classes predicted by denoising network
			pE: edge classes
		"""
		return self.get_network_predictions(graph, key)

	#-------------------------------------------------------------------

	def denoise(self, graph: Graph, t: int, key: jax.Array)->Graph:

		kP, kX, kE = jr.split(key, 3)
		
		pX, pE = self.get_sampling_probas(graph, t, kP)

		X = jnn.one_hot(jr.categorical(kX, pX), num_classes=pX.shape[-1])
		E = jnn.one_hot(jr.categorical(kE, pE), num_classes=pE.shape[-1])
		E = E.reshape((graph.N, graph.N, -1))

		return eqx.tree_at(lambda G: [G.nodes.h, G.edges.e],
						   graph,
						   [X, E])

	#-------------------------------------------------------------------

	def get_sampling_probas(self, graph: Graph, t: int, key: jax.Array)->Tuple[jax.Array, jax.Array]:
		"""
		Computes P(X(t-1)|G(t)) and P(E(t-1)|G(t))
		"""
		key, net_key = jr.split(key)
		Xt, Et = graph.h, graph.edges.e
		assert Et is not None
		pX, pE = self.denoising_network(graph, net_key)
		X_post, E_post = self.noise_model.get_posterior(Xt, Et, t)

		wX = pX[..., None] * X_post
		unnorm_pX = wX.sum(1)
		unnorm_pX = jnp.where(unnorm_pX.sum(-1, keepdims=True)==0, 1e-6, unnorm_pX)
		pX = unnorm_pX / jnp.sum(unnorm_pX, axis=-1, keepdims=True) #type:ignore

		wE = pE[..., None] * E_post
		unnorm_pE = wE.sum(1)
		unnorm_pE = jnp.where(unnorm_pE.sum(-1, keepdims=True)==0, 1e-6, unnorm_pE)
		pE = unnorm_pE / jnp.sum(unnorm_pE, axis=-1, keepdims=True) #type:ignore
		
		return pX, pE

	#-------------------------------------------------------------------
	
	def sample_denoising_trajectory(self, graph: Graph, T: int, key: jax.Array)->Graph:

		def _denoise_step(carry, t):
			G, key = carry
			key, skey = jr.split(key)
			G = self.denoise(graph, t, skey)
			return [G, key], G
		_, Gs = jax.lax.scan(_denoise_step, [graph, key], jnp.arange(T))
		return Gs

	#-------------------------------------------------------------------

	def sample_noisy_graph(self, graph: Graph, t: Int, key: jax.Array)->Graph:

		return self.noise_model.sample_noisy_graph(graph, t, key)

	#-------------------------------------------------------------------

	def get_network_predictions(self, graph: Graph, key: jax.Array):
		
		pX, pE = self.denoising_network(graph, key)
		pG = make_graph(pX, pE)
		pX, pE = self.order_graph(pG)
		return pX, pE

	#-------------------------------------------------------------------

	def order_graph(self, graph: Graph):

		assert graph.E is not None

		if self.ordering == "degree":
			degs = degrees(graph.A)
			order = jnp.argsort(degs)
		elif self.ordering=="data":
			order = jnp.arange(graph.N)
		else:
			warn(f"ordering {self.ordering} is not a valid ordering scheme. Using default data ordering instead")
			order = jnp.arange(graph.N)

		return graph.X[order], graph.E[order][:, order]

	#-------------------------------------------------------------------




if __name__ == '__main__':

	from src.models.diffusion.denoising import DummyDenoisingNetwork, GraphTransformerDenoisingNetwork
	from src.gnn.utils import viz_graph
	import matplotlib.pyplot as plt
	
	k = jr.PRNGKey(1)
	N = 4
	nX = 2
	nE = 2
	X = jr.normal(k, (N, nX))
	X = jnn.one_hot(jnp.argmax(X, axis=-1), nX)
	E = jr.normal(k, (N, N, nE))
	E = jnn.one_hot(jnp.argmax(E, axis=-1), nE)
	A = E[:, :, 1]

	G = Graph(nodes=Node(h=X), edges=Edge(e=E, A=A))

	noise_model = DiscreteNoiseModel(nX, nE, 30)
	nh = 2
	dqk = 3
	dv = 5
	denoising_net = GraphTransformerDenoisingNetwork(
		1, nX, nE, nX, nE, nh, dqk, dv, 1, 3, 1, 3, True, False, False, key=k
	)

	model = DiffusionModel(noise_model, denoising_net)
	print(G.E[...,1])
	t=2
	k, k_ = jr.split(k)
	Gt = model.sample_noisy_graph(G, t, k_)
	Gt = eqx.tree_at(lambda g: g.nodes.m, Gt, X[:,1], is_leaf=lambda x: x is None)
	print(Gt.E[...,1])


	






