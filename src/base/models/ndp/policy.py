from functools import partial
from typing import Callable, Tuple, Union, Optional, NamedTuple
from jax._src.lax.utils import _input_dtype
from jaxtyping import Float, PyTree
from numpy import partition
from src.gnn.base import Graph
from src.models.ndp.ndp import NDP

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn

class RNNPolicyState(NamedTuple):
	w: jax.Array
	h: jax.Array
	m: jax.Array
	a: Optional[jax.Array]=None

class RNNPolicy(eqx.Module):
	#-------------------------------------------------------------------
	action_dims: int
	obs_dims: int
	dev_steps: int
	policy_iters: int
	ndp: NDP
	is_recurrent: bool
	input_mode: str
	discrete_action: bool
	take_last_for_action: bool
	#-------------------------------------------------------------------

	def __init__(self, action_dims: int, obs_dims: int, ndp: NDP, policy_iters: int, 
				 dev_steps: int, is_recurrent: bool=True, input_mode: str="set",
				 action_is_discrete: bool=False, take_last_for_action: bool=False,
				 *, key: jax.Array):

		self.action_dims=action_dims
		self.obs_dims = obs_dims
		self.policy_iters = policy_iters
		self.dev_steps = dev_steps
		self.ndp = ndp
		self.is_recurrent = is_recurrent
		self.input_mode = input_mode
		self.discrete_action = action_is_discrete
		self.take_last_for_action = take_last_for_action

	#-------------------------------------------------------------------

	def __call__(self, obs: jax.Array, state: RNNPolicyState, key: jax.Array)->Tuple[jax.Array, RNNPolicyState]:

		def rnn_step(h):
			w = state.w * (state.m[:, None]*state.m[None])
			return jnn.tanh(h @ w)
		if self.input_mode == "set":
			h = state.h.at[:self.obs_dims].set(obs)
		elif self.input_mode == "add":
			h = state.h.at[:self.obs_dims].add(obs)
		else: 
			raise NotImplementedError(f"input mode {self.input_mode} is not a valid mode, use either 'add' or 'set'")
		h = jax.lax.fori_loop(0, self.policy_iters, lambda _, h: rnn_step(h), h)
		if state.a is None:
			N = state.m.sum().astype(int)
			start = (N - self.action_dims,)
			size = (self.action_dims,)
			a = jax.lax.dynamic_slice(h, start, size)
		else:
			a = jnp.take(h, state.a)
		if self.discrete_action:
			a = jnp.argmax(a)
		if self.is_recurrent:
			state = state._replace(h=h)
		return a, state

	#-------------------------------------------------------------------

	def initialize(self, key: jax.Array)->RNNPolicyState:
		G = self.ndp.init_and_rollout_(key, self.dev_steps)
		w = G.edges.e
		if len(w.shape)==3:
			w = w[...,0]
		w = w * G.A
		m = G.nodes.m
		h = jnp.zeros((G.N,))
		if self.take_last_for_action:
			a = None
		else:
			_, a = jax.lax.top_k(jnp.where(m.astype(bool), G.h[..., 0], -jnp.inf), self.action_dims) #type:ignore
		return RNNPolicyState(
			w = w,
			h = h,
			m = m,
			a = a
		)

#=======================================================================
#=======================================================================

class HebbianPolicyState(NamedTuple):
	w: jax.Array
	h: jax.Array
	eta: Float
	A: Float
	B: Float
	C: Float
	D: Float
	Am: jax.Array
	m: Optional[jax.Array]=None
	a: Optional[jax.Array]=None

class HebbianRNN(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	# Parameters:
	ndp: NDP
	# Statics:
	dev_steps: int
	random_weights: bool
	discrete_action: bool
	action_dims: int
	take_last_for_action: bool
	#-------------------------------------------------------------------

	def __init__(self, ndp: NDP, action_dims: int, dev_steps: int, random_weights: bool=True, 
				 discrete_action: bool=False, take_last_for_action: bool=False):
		
		self.ndp = ndp
		self.dev_steps = dev_steps
		self.random_weights = random_weights
		self.discrete_action = discrete_action
		self.action_dims = action_dims
		self.take_last_for_action = take_last_for_action

	#-------------------------------------------------------------------

	def __call__(self, obs: jax.Array, state: HebbianPolicyState, key: jax.Array)->Tuple[jax.Array, HebbianPolicyState]:
		
		h = state.h.at[:obs.shape[0]].set(obs)
		h = jnn.tanh(state.w @ h)
		
		hh = h[None] * h[:, None]
		dw = state.eta * (state.A * hh + state.B*h[None] + state.C*h[:,None] + state.D)

		w = (state.w + dw) * state.Am

		if state.a is not None:
			a = jnp.take(h, state.a)
		else:
			assert state.m is not None
			N = state.m.sum().astype(int)
			start = (N - self.action_dims,)
			size = (self.action_dims,)
			a = jax.lax.dynamic_slice(h, start, size)

		return a, state._replace(h=h, w=w), 

	#-------------------------------------------------------------------

	def initialize(self, key: jax.Array)->HebbianPolicyState:

		ndp_key, w_key = jr.split(key)
		G = self.ndp.init_and_rollout_(ndp_key, self.dev_steps)
		if self.random_weights:
			w = jr.uniform(w_key, (G.N, G.N), minval=-.1, maxval=.1) * G.A
			eta, A, B, C, D = G.e[:, :, :5].transpose((2,0,1))
		else:
			w = G.e[..., 0] if len(G.e.shape)==3 else G.e
			w = w* G.A
			eta, A, B, C, D = G.e[..., 1:6].transpose((2,0,1))
		if self.take_last_for_action:
			a = None
		else:
			_, a = jax.lax.top_k(jnp.where(m.astype(bool), G.h[..., 0], -jnp.inf), self.action_dims) #type:ignore
		h = jnp.zeros((G.N,))
		return HebbianPolicyState(w=w, h=h, m=G.nodes.m, eta=eta,
								  A=A, B=B, C=C, D=D, Am=G.A, a=a)


#=======================================================================
#=======================================================================

class NeuroDiversePolicyState(NamedTuple):
	h: jax.Array
	m: jax.Array
	n: jax.Array
	x: jax.Array
	w: jax.Array


class NeuroDiversePolicy(eqx.Module):
	"""
	TODO
	https://arxiv.org/pdf/2305.15945.pdf
	"""
	#-------------------------------------------------------------------
	# Parameters:
	ndp: NDP
	# Statics:
	action_dims: int
	hidden_features: int
	dev_steps: int
	discrete_action: bool
	#-------------------------------------------------------------------

	def __init__(self, ndp: NDP, hidden_features: int, action_dims: int, dev_steps: int, 
				 discrete_action: bool=False, *, key: jax.Array) :
		
		self.ndp = ndp
		self.hidden_features = hidden_features
		self.dev_steps = dev_steps
		self.discrete_action = discrete_action
		self.action_dims = action_dims

	#-------------------------------------------------------------------

	def __call__(self, obs: jax.Array, state: NeuroDiversePolicyState, key: jax.Array)->Tuple[jax.Array, NeuroDiversePolicyState]:
		
		x = state.w @ state.x.at[:, :obs.shape[0]].set(obs)
		x = jnp.stack([state.h, x, jnp.ones_like(state.h)], axis=1)
		hy = jnn.tanh(jax.vmap(jnp.dot)(x, state.n))
		h, y = jnp.split(hy, 2, axis=-1)
		N = state.m.sum().astype(int)
		start = (N - self.action_dims,)
		size = (self.action_dims,)
		a = jax.lax.dynamic_slice(y, start, size)
		if self.discrete_action:
			a = jnp.argmax(a)

		return a, state._replace(x=y, h=h)

	#-------------------------------------------------------------------
	
	def initialize(self, key: jax.Array)->NeuroDiversePolicyState:
		
		G = self.ndp.initialize(key)
		h = jnp.zeros((G.N,))
		n = G.h.reshape((G.N, self.hidden_features+2, self.hidden_features+1))
		m = G.nodes.m
		x = jnp.zeros((G.N))
		w = G.e if len(G.e.shape)==2 else G.e[...,0] #type:ignore
		return NeuroDiversePolicyState(h=h, m=m, n=n, x=x, w=w) #type:ignore

	#-------------------------------------------------------------------

	def partition(self):
		return eqx.partition(self, eqx.is_array)





