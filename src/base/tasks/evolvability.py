from typing import Callable, Optional
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
from src.tasks.base import QDTask, Params, Data

class EvolvabilityTask(eqx.Module):
	
	"""
	Wrapper transforming QD task into evolvability task
	i.e fitness is combimation of perf + variation of descriptors
	"""
	#-------------------------------------------------------------------
	task: QDTask
	n_childs: int
	noise_generator: Callable
	#-------------------------------------------------------------------

	def __init__(self, task: QDTask, n_childs: int, noise_generator: Callable):
		""""""
		self.task = task
		self.n_childs = n_childs
		self.noise_generator = noise_generator

	#-------------------------------------------------------------------

	def __call__(self, params: Params, key: jax.Array, data: Optional[Data]=None):
		""""""
		def _eval(params, key, data):
			noise_key, eval_key = jr.split(key)
			noise = self.noise_generator(noise_key)
			noisy_params = jax.tree_map(lambda t1, t2: t1+t2, params, noise)
			return self.task(noisy_params, eval_key, data)
		keys = jr.split(key, self.n_childs)
		return jax.vmap(_eval, in_axes=(None, 0, None))(params, keys, data)

		