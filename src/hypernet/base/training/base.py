from hypernet.training.utils import progress_bar_scan, progress_bar_fori

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import Callable, NamedTuple, Optional, Tuple, Any, TypeAlias
import jax.experimental.host_callback as hcb
import wandb

from jaxtyping import PyTree

Data: TypeAlias = PyTree[...]
TaskParams: TypeAlias = PyTree[...]
TrainState: TypeAlias = PyTree[...]

class BaseTrainer(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	train_steps: int
	wandb_log: Optional[bool]
	metrics_fn: Optional[Callable]
	progress_bar: Optional[bool]
	#-------------------------------------------------------------------

	def __init__(self, train_steps: int, 
				 wandb_log: Optional[bool]=False, 
				 metrics_fn: Optional[Callable]=None,
				 progress_bar: Optional[bool]=False):
		
		self.train_steps = train_steps
		self.wandb_log = wandb_log
		self.metrics_fn = metrics_fn
		self.progress_bar = progress_bar

	#-------------------------------------------------------------------

	def __call__(self, key: jr.PRNGKeyArray):

		return self.init_and_train(key)

	#-------------------------------------------------------------------

	def train(self, state: TrainState, key: jax.Array, task_params: Optional[TaskParams]=None)->Tuple[TrainState, Data]:

		def _step(c, x):
			s, k = c
			k, k_ = jr.split(k)
			s, data = self.train_step(s, k_)
			
			if self.metrics_fn is not None:
				data = self.metrics_fn(s, data)
			if self.wandb_log:
				self.log(data)

			return [s, k], {"states": s, "metrics": data}

		if self.progress_bar:
			_step = progress_bar_scan(self.train_steps)(_step) #type: ignore

		[state, key], data = jax.lax.scan(_step, [state, key], jnp.arange(self.train_steps))

		return state, data

	#-------------------------------------------------------------------

	def train_(self, state: TrainState, key: jax.Array, task_params: Optional[TaskParams]=None)->TrainState:

		def _step(i, c):
			s, k = c
			k, k_ = jr.split(k)
			s, data = self.train_step(s, k_)
			
			if self.metrics_fn is not None:
				data = self.metrics_fn(s, data)
			if self.wandb_log:
				self.log(data)

			return [s, k]

		if self.progress_bar:
			_step = progress_bar_fori(self.train_steps)(_step) #type: ignore

		[state, key] = jax.lax.fori_loop(0, self.train_steps, _step, [state, key])

		return state

	#-------------------------------------------------------------------

	def log(self, data):
		hcb.id_tap(
			lambda d, *_: wandb.log(d), data
		)

	#-------------------------------------------------------------------

	def init_and_train(self, key: jr.PRNGKeyArray, task_params: Optional[TaskParams]=None)->Tuple[TrainState, Data]:
		init_key, train_key = jr.split(key)
		state = self.initialize(init_key)
		return self.train(state, train_key, task_params)

	#-------------------------------------------------------------------

	def init_and_train_(self, key: jr.PRNGKeyArray, task_params: Optional[TaskParams]=None)->TrainState:
		init_key, train_key = jr.split(key)
		state = self.initialize(init_key)
		return self.train_(state, train_key, task_params)

	#-------------------------------------------------------------------

	def train_step(self, state: TrainState, key: jr.PRNGKeyArray, task_params: Optional[TaskParams]=None)->Tuple[TrainState, Any]:
		raise NotImplementedError

	#-------------------------------------------------------------------

	def initialize(self, key: jr.PRNGKeyArray)->TrainState:
		raise NotImplementedError