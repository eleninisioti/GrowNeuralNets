from chex import PRNGKey
from src.gnn.base import GraphModule, Graph, Node, Edge

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn


class FiLM(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	# Parameters:
	W1: jax.Array
	W2: jax.Array
	# Statics:
	#-------------------------------------------------------------------

	def __init__(self, N: int, *, key: jax.Array):
		
		key_w1, key_w2 = jr.split(key)
		self.W1 = jr.normal(key_w1, (N, N))
		self.W2 = jr.normal(key_w2, (N, N))

	#-------------------------------------------------------------------

	def __call__(self, M1: jax.Array, M2: jax.Array)->jax.Array:
		y = jnp.dot(M1, self.W1) + (jnp.dot(M1, self.W2) * M2) + M2
		return y

def scaled_dot_product(Q, K):
	return jnp.dot(Q, K.T)

class DiGressTransformer(GraphModule):
	
	"""
	"""
	#-------------------------------------------------------------------
	WQ: nn.Linear
	WK: nn.Linear
	WV: nn.Linear
	E_mul: nn.Linear
	E_add: nn.Linear
	Xout: nn.Linear
	Eout: nn.Linear
	ffE: nn.MLP
	ffX: nn.MLP
	H: int
	#-------------------------------------------------------------------

	def __init__(self, N: int, dx: int, dk: int, dv: int, de: int,  n_heads: int, *, key: jax.Array):
		
		key_Q, key_K, key_V, key_add, key_mul, key_Xout, key_Eout, key_ffX, key_ffE = jr.split(key, 7)
		
		self.WQ = nn.Linear(dx, dk*n_heads, key=key_Q)
		self.WK = nn.Linear(dx, dk*n_heads, key=key_K)
		self.WV = nn.Linear(dx, dv*n_heads, key=key_V)
		self.E_mul = nn.Linear(de, dk*n_heads, key=key_mul)
		self.E_add = nn.Linear(de, dk*n_heads, key=key_add)
		self.Xout = nn.Linear(dv, dx, key=key_Xout)
		self.Eout = nn.Linear(dk*n_heads, de, key=key_Eout)
		self.ffX = nn.MLP(dx, dx, 64, 2, key=key_ffX)
		self.ffE = nn.MLP(de, de, 64, 2, key=key_ffE)

		self.H = n_heads

	#-------------------------------------------------------------------

	def apply_adj(self, graph: Graph, key: jr.PRNGKeyArray) -> Graph:
		assert graph.edges.e is not None
		
		X, E, N = graph.h, graph.edges.e, graph.N

		Q = jax.vmap(self.WQ)(X).reshape((N, self.H, -1)) # N x H x dk
		K = jax.vmap(self.WK)(X).reshape((N, self.H, -1)) # N x H x dk
		V = jax.vmap(self.WV)(X).reshape((N, self.H, -1)) # N x H x dv

		Y = Q[None,...] * K[:,None,...] # N x N x H x dk

		E_mul = jax.vmap(jax.vmap(self.E_mul))(E).reshape((N,N,self.H,-1))
		E_add = jax.vmap(jax.vmap(self.E_add))(E).reshape((N,N,self.H,-1))

		Y = (Y * (E_mul+1) + E_add) # N x N x H x dk
		att = jnn.softmax(Y.sum(-1), axis=1)
		wV = (V[None,...]*att[...,None]).sum((1,2))
		dX = jax.vmap(self.Xout)(wV)
		dE = jax.vmap(jax.vmap(self.Eout))(Y.reshape((n,n,-1)))

		X = X + dX
		X = X / (jnp.linalg.norm(X, axis=-1)+1e-8)

		E = E + dE
		E = E / (jnp.linalg.norm(E, axis=-1)+1e-8)

		dX = jax.vmap(self.ffX)(X)
		dE = jax.vmap(jax.vmap(self.ffE))(E)

		X = X + dX
		X = X / (jnp.linalg.norm(X, axis=-1)+1e-8)

		E = E + dE
		E = E / (jnp.linalg.norm(E, axis=-1)+1e-8)

		return eqx.tree_at(
			lambda G: [G.nodes.h, G.edges.e],
			graph,
			[X, E]
		)



if __name__ == '__main__':
	n = 10
	dx = 4
	de = 3
	dk = 2
	dv = 9
	h = 5
	f = DiGressDenoising(n, dx, dk, dv, de, h, key=jr.PRNGKey(1))

	nodes = Node(h= jnp.ones((n, dx)))
	edges = Edge(A=jnp.zeros((n,n)), e=jnp.ones((n, n, de)))
	G = Graph(nodes=nodes, edges=edges)

	f(G, jr.PRNGKey(1))














