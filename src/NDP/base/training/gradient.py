from src.training.base import BaseTrainer

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import optax
from typing import Callable, Optional, Union, Tuple, NamedTuple
from jaxtyping import Float, PyTree

Params = PyTree[...]
Data = PyTree[...]

class TrainState(NamedTuple):
	params: Params
	opt_state: optax.OptState

class OptaxTrainer(BaseTrainer):
	
	"""
	"""
	#-------------------------------------------------------------------
	optimizer: optax.GradientTransformation
	loss_fn: Callable[[Params, jr.PRNGKeyArray], Tuple[Float, Data]]
	initializer: Callable[[jr.PRNGKeyArray], Params]
	#-------------------------------------------------------------------

	def __init__(
		self, 
		train_steps: int,
		optimizer: Union[optax.GradientTransformation, str],
		initializer: Callable[[jr.PRNGKeyArray], Params],
		loss_fn: Callable[[Params, jr.PRNGKeyArray], Float], 
		learning_rate: Optional[float]=0.01,
		opt_kws: Optional[dict]={},
		wandb_log: Optional[bool]=False, 
	    metrics_fn: Optional[Callable]=None,
	    progress_bar: Optional[bool]=False):

		super().__init__(train_steps, wandb_log=wandb_log, metrics_fn=metrics_fn, progress_bar=progress_bar)
		
		if isinstance(optimizer, str):
			OPT = getattr(optax, optimizer)
			self.optimizer = OPT(learning_rate=learning_rate, **opt_kws)
		else:
			self.optimizer = optimizer

		self.loss_fn = loss_fn
		self.initializer = initializer

	#-------------------------------------------------------------------

	def train_params(self, params: Params, key: jr.PRNGKeyArray)->Tuple[TrainState, Data]:

		state = TrainState(params=params, opt_state=self.optimizer.init(params))
		return self.train(state, key) # type: ignore

	#-------------------------------------------------------------------

	def train_step(self, state: TrainState, key: jr.PRNGKeyArray) -> Tuple[TrainState, Data]:
		
		[loss, eval_data], grads = jax.value_and_grad(self.loss_fn, has_aux=True)(state.params, key)
		updates, opt_state = self.optimizer.update(grads, state.opt_state)
		params = optax.apply_updates(state.params, updates)
		return TrainState(params=params, opt_state=opt_state), {"loss": loss, "eval_data": eval_data}

	#-------------------------------------------------------------------

	def initialize(self, key: jr.PRNGKeyArray) -> TrainState:
		
		init_params = self.initializer(key)
		opt_state = self.optimizer.init(init_params)
		return TrainState(params=init_params, opt_state=opt_state)