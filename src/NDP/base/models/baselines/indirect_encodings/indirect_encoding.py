from typing import Callable, NamedTuple, Optional
from src.models.base import BaseModel

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn
from jaxtyping import Array, Float, PyTree

class CPPN(BaseModel):
	
	"""
	"""
	#-------------------------------------------------------------------
	net: PyTree[...]
	#-------------------------------------------------------------------

	def __init__(
		self, 
		neuron_features: int = 2,
		output_features: int = 1,
		aux_features: int=0,
		width: int = 2,
		depth: int = 2, 
		activation: Callable = jnn.tanh, 
		final_activation: Callable = jnn.sigmoid,
		*,
		key: jr.PRNGKeyArray):
		
		self.net = nn.MLP(neuron_features*2 + aux_features, output_features, 
			 			  width, depth, activation=activation, 	
						  final_activation=final_activation, key=key)

	#-------------------------------------------------------------------

	def __call__(self, 
		x1: Float[jax.Array, "N neuron_features"], 
		x2: Float[jax.Array, "N neuron_features"]):
		
		return jax.vmap(self.net)(jnp.concatenate([x1, x2], axis=-1))


class DNA_CPPN(CPPN):
	
	"""
	"""
	#-------------------------------------------------------------------
	dna: Float[Array, "D_dna"]
	#-------------------------------------------------------------------

	def __init__(
		self, 
		neuron_features: int = 2,
		output_features: int = 1,
		dna_features: int=0,
		width: int = 2,
		depth: int = 2, 
		activation: Callable = jnn.tanh, 
		final_activation: Callable = jnn.sigmoid,
		*,
		key: jr.PRNGKeyArray):

		self.dna = jr.normal(key, (dna_features, ))
		super().__init__(neuron_features, output_features, dna_features, 
						 width, depth, activation, final_activation, key=key)

	#-------------------------------------------------------------------
		
	def __call__(
		self, 
		x1: Float[jax.Array, "N neuron_features"], 
		x2: Float[jax.Array, "N neuron_features"]):
		
		return jax.vmap(self.net)(jnp.concatenate([x1, x2, self.dna[None,:]], axis=-1))

	#-------------------------------------------------------------------

	def dna_partition(self):
		params, statics = eqx.partition(self, eqx.is_array)
		spec = jax.tree_map(lambda x: True, params)
		spec = eqx.tree_at(lambda tree: tree.dna, False)
		params, dparams = eqx.partition(params, spec)
		return params, dparams, statics


class GEM(BaseModel):
	
	"""
	https://www.nature.com/articles/s41467-023-37980-1
	"""
	#-------------------------------------------------------------------
	Xinp: Float[Array, "Dinp G"]
	Xh: Float[Array, "Depth Dh G"]
	Xout: Float[Array, "Dout G"]
	O: Float[Array, "G G"]
	depth: int
	#-------------------------------------------------------------------

	def dna_partition(self):
		params, statics = eqx.partition(self, eqx.is_array)
		spec = jax.tree_map(lambda x: True, params)
		spec = eqx.tree_at(lambda tree: [tree.Xinp, tree.Xout, tree.Xh], [False, False, False])
		params, dparams = eqx.partition(params, spec)
		return params, dparams, statics

	#-------------------------------------------------------------------

	def __init__(self, 
		n_genes: int,
		input_dims: int,
		output_dims: int,
		width: int,
		depth: int,
		*,
		key: jr.PRNGKeyArray):
		
		key, key_Xh, key_Xout, key_Xinp, key_O = jr.split(key, 3)
		
		self.Xinp = jr.normal(key_Xinp, (input_dims, n_genes))
		self.Xout = jr.normal(key_Xout, (output_dims, n_genes))
		self.Xh = jr.normal(key_Xh, (depth, width, n_genes))
		self.O = jr.normal(key_O, (n_genes, n_genes))
		self.depth = depth

	#-------------------------------------------------------------------

	def __call__(self):

		def get_W(Xi, Xo):
			return Xi @ self.O @ Xo.T
		
		Wih = get_W(self.Xinp, self.Xh[0])
		Who = get_W(self.Xh[-1], self.Xout)

		if self.depth > 1:
			Whh = jax.vmap(get_W)(self.Xh[:-1], self.Xh[1:])
			return [Wih, Whh, Who]
		return [Wih, Who]



class S_GEMState(NamedTuple):
	p_inp: Float[Array, "Dinp X"]
	p_h: Float[Array, "Depth Dh X"]
	p_out: Float[Array, "Dout X"]


def gaussian(x, mu, sigma):
	return jnp.exp(- jnp.sqrt(jnp.square(x-mu).sum()) / (2*sigma**2))


class S_GEM(BaseModel):
	
	"""
	https://www.nature.com/articles/s41467-023-37980-1
	"""
	#-------------------------------------------------------------------
	mu: Float[Array, "G X"]
	sigma: Float[Array, "G"]
	O: Float[Array, "G G"]
	#-------------------------------------------------------------------

	def __init__(self, args):
		pass

	#-------------------------------------------------------------------

	def __call__(self):
		pass

