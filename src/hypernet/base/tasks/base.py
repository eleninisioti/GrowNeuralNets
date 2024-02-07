from abc import ABC
from typing import Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
from jaxtyping import Float, PyTree

TaskState = PyTree[...]
TaskParams = PyTree[...]
Params = PyTree[...]
Data = PyTree[...]

class BaseTask(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	#-------------------------------------------------------------------

	def __call__(self, params: Params, key:jr.PRNGKeyArray, task_params: TaskParams=None)->Tuple[Float, Data]:
		
		raise NotImplementedError

	#-------------------------------------------------------------------


