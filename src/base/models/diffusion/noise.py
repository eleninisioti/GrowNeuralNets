import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
from src.gnn.base import Graph, Node, Edge
import equinox as eqx
from typing import Optional, Tuple
from jaxtyping import Int


def sample_categorical(p: jax.Array, key: jax.Array):
	cum_prob = jnp.cumsum(p)
	r = jr.uniform(key)
	return jnp.argmax(cum_prob>r)


class NoiseModel(eqx.Module):
	"""
	"""
	#-------------------------------------------------------------------
	#-------------------------------------------------------------------
	def __call__(self, graph: Graph, key: jax.Array)->Graph:
		raise NotImplementedError
	#-------------------------------------------------------------------
	def get_diffusion_trajectory(self, init_graph: Graph, diffusion_steps: int, key: jax.Array)->Graph:

		def _diffuse(G, key):
			nG = self.__call__(G, key)
			return nG, G
		_, Gs = jax.lax.scan(_diffuse, init_graph, jr.split(key, diffusion_steps))
		return Gs
	#-------------------------------------------------------------------
	def sample_noisy_graph(self, graph: Graph, t: int, key: jax.Array)->Graph:
		raise NotImplementedError
	#-------------------------------------------------------------------
	def get_posterior(self, Xt: jax.Array, Et: jax.Array, t: int):
		raise NotImplementedError
	#-------------------------------------------------------------------


class DiscreteNoiseModel(NoiseModel):
	"""
	Categorical noise model defined by transition matrices between classes
	"""
	#-------------------------------------------------------------------
	QX: jax.Array # node clas stransition matrix
	QE: jax.Array # edge class transition matrix
	mode: str # noise mode in {"uniform", "marginal"}
	NX: int
	NE: int
	T: Int
	noise_schedule: str
	#-------------------------------------------------------------------

	def __init__(self, NX: int, NE: int, T: Int, mode: str="uniform",
				 mX: Optional[jax.Array]=None, mE: Optional[jax.Array]=None,
				 noise_schedule: str="cosine"):
		
		self.mode = mode
		self.NX = NX
		self.NE = NE
		self.T = T
		self.noise_schedule = noise_schedule
		if mode == "uniform":
			self.QX = jnp.ones((NX, NX)) / NX
			self.QE = jnp.ones((NE, NE)) / NE
		elif mode == "marginal":
			assert mX is not None and mE is not None
			self.QX = jnp.repeat(mX[None, :], NX, axis=0)
			self.QE = jnp.repeat(mE[None, :], NE, axis=0)
		else:
			raise NotImplementedError(f"mode {mode} not NotImplementedError")

	#-------------------------------------------------------------------

	def sample_noisy_graph(self, graph: Graph, t: int, key: jax.Array) -> Graph:
		"""
		Sample noisy graph at time t=t from clean graph (t=0)
		"""
		assert graph.edges.e is not None

		qX, qE = self.get_Qt_bar(t)
		X, E = graph.h, graph.edges.e
		N = X.shape[0]
		pX = graph.h @ qX
		pE = E.reshape((-1, self.NE)) @ qE.reshape((-1, self.NE))

		kX, kE = jr.split(key)
		X = jax.vmap(sample_categorical)(pX, jr.split(kX, pX.shape[0]))
		X = jnn.one_hot(X, self.NX)

		E = jax.vmap(sample_categorical)(pE, jr.split(kE, pE.shape[0]))
		E = jnn.one_hot(E, self.NE).reshape((N, N, self.NE))

		A = E[:, :, 1]

		return eqx.tree_at(lambda G: [G.nodes.h, G.edges.e, G.edges.A],
						   graph,
						   [X, E, A])

	#-------------------------------------------------------------------

	def get_Qt(self, t: int)->Tuple[jax.Array, jax.Array]:
		"""returns one-step transition matrices between t-1 and t"""
		
		if self.noise_schedule == "cosine":
			halfpi = 0.5*jnp.pi
			s = .01
			alpha_t = jnp.cos(halfpi*(t/self.T+s)/(1+s))
		else:
			raise ValueError(f"noise schedule {self.noise_schedule} is not a valid schedule")
		qxt = alpha_t * self.QX + (1 - alpha_t) * jnp.eye(self.QX.shape[0])
		qet = alpha_t * self.QE + (1 - alpha_t) * jnp.eye(self.QE.shape[0])
		return qxt, qet

	#-------------------------------------------------------------------

	def get_Qt_bar(self, t: int):
		"""returns transition matrices between step 0 and step t"""
		if self.noise_schedule == "cosine":
			halfpi = 0.5*jnp.pi
			s = .01
			alpha_bar_t = jnp.cos(halfpi*(t/self.T+s)/(1+s))
		else:
			raise ValueError(f"noise schedule {self.noise_schedule} is not a valid schedule")
		qxt = alpha_bar_t * self.QX + (1 - alpha_bar_t) * jnp.eye(self.QX.shape[0])
		qet = alpha_bar_t * self.QE + (1 - alpha_bar_t) * jnp.eye(self.QE.shape[0])
		return qxt, qet

	#-------------------------------------------------------------------

	def get_posterior(self, Xt: jax.Array, Et: jax.Array, t: int):

		if len(Et.shape)==3:
			Et = Et.reshape((-1, Et.shape[-1]))

		qxt, qet = self.get_Qt(t)
		qxbs, qebs = self.get_Qt_bar(t-1)

		X_left = (Xt @ qxt.T)[:, None, :]
		E_left = (Et @ qet.T)[:, None, :]

		X_right = qxbs[None,...]
		E_right = qebs[None,...]

		X_num = X_left * X_right
		E_num = E_left * E_right

		X_prod = (qxt @ Xt.T).T[...,None]
		E_prod = (qet @ Et.T).T[...,None]

		X_den = jnp.where(X_prod==0., 1e-6, X_prod)
		E_den = jnp.where(E_prod==0., 1e-6, E_prod)

		return X_num/X_den, E_num/E_den



if __name__ == '__main__':
	
	key = jr.PRNGKey(1)
	N = 5
	X = jnn.softmax(jr.normal(key, (N, 3)), axis=-1)
	E = jnn.softmax(jr.normal(key, (N, N, 2)), axis=-1)
	A = jnp.zeros((N, N))

	G = Graph(nodes=Node(h=X), edges=Edge(A=A, e=E))

	noise_model = DiscreteNoiseModel(3, 2, .8, 30, mode="uniform")

		




