from jax._src.core import Value
from src.tasks.base import BaseTask, TaskParams

from typing import Callable, Iterable, Union, Tuple
from jaxtyping import Float, PyTree
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx

Data = PyTree
Params = PyTree

stack_trees = lambda trees: jax.tree_map(lambda ts: jnp.stack(trees), trees)

class MultiTask(BaseTask):
	
	"""
	"""
	#-------------------------------------------------------------------
	tasks: Iterable[Union[BaseTask, Callable]]
	mode: str
	#-------------------------------------------------------------------

	def __init__(self, tasks: Iterable[Union[BaseTask, Callable]], mode: str= "all"):
		
		assert mode in ["all"], f"mode {mode} is not supported or not a valid mode"
		self.tasks = tasks
		self.mode = mode

	#-------------------------------------------------------------------

	def __call__(self, params: Params, key: jax.Array, task_params: TaskParams = None) -> Tuple[Float, Data]:
		
		if self.mode == "all":
			return self._eval_all(params, key, task_params)
		else: 
			raise ValueError(f"mode {self.mode} is not supported or not a valid mode")

	#-------------------------------------------------------------------

	def _eval_all(self, params: Params, key: jax.Array, task_params: TaskParams = None) -> Tuple[Float, Data]:
		"""evaluate params on all tasks """
		datas = []
		fit_sum = 0.
		for task in self.tasks:
			key, subkey = jr.split(key)
			fit, data = task(params, subkey, task_params)
			fit_sum = fit_sum + fit
			datas.append(data)
		return fit_sum, stack_trees(datas)