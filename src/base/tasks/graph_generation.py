from src.tasks.base import *

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx


class GraphFeatureGeneration(BaseTask):

	#-------------------------------------------------------------------

	def __init__(self, statics, target_size, target_density, dev_steps):
		
		self.target = jnp.array([target_size, target_density])
		self.statics = statics
		self.dev_steps = dev_steps

	#-------------------------------------------------------------------

	def __call__(self, params: Params, key: jr.PRNGKeyArray, data: Optional[Data] = None) -> Tuple[Float, Data]:

		model = eqx.combine(params, self.statics)
		G = model.init_and_rollout_(key)
		size = G.nodes.m.sum()
		density = G.A.sum() / size**2

		f = jnp.array([size, density])

		error = jnp.mean(jnp.square(f-self.target))

		return -error, None





