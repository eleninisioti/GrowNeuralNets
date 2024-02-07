from NDP.training.base import BaseTrainer

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import evosax as ex
import equinox as eqx
from typing import Any, Callable, Collection, Dict, Optional, Union, NamedTuple, Tuple
from jaxtyping import Array, Float, PyTree

TrainState = ex.EvoState
Params = PyTree[...]
Data = PyTree[...]
TaskParams = PyTree[...]


def default_metrics(state, data):
	y = {}

	y["best"] = state.best_fitness
	y["gen_best"] = data["fitness"].min()
	y["gen_mean"] = data["fitness"].mean()
	y["gen_worse"] = data["fitness"].max()

	return y


class EvosaxTrainer(BaseTrainer):
	
	"""
	"""
	#-------------------------------------------------------------------
	strategy: ex.Strategy
	es_params: ex.EvoParams
	params_shaper: ex.ParameterReshaper
	task: Callable
	fitness_shaper: ex.FitnessShaper
	#-------------------------------------------------------------------

	def __init__(
		self, 
		train_steps: int,
		strategy: Union[ex.Strategy, str],
		task: Callable,
		params_shaper: ex.ParameterReshaper,
		popsize: Optional[int]=None,
		fitness_shaper: Optional[ex.FitnessShaper]=None,
		es_kws: Optional[Dict[str, Any]]={},
		es_params: Optional[ex.EvoParams]=None,
		eval_reps: int=1,
		wandb_log: Optional[bool]=False, 
	    metrics_fn: Optional[Callable]=None,
	    progress_bar: Optional[bool]=False):

		super().__init__(train_steps, wandb_log=wandb_log, metrics_fn=metrics_fn, progress_bar=progress_bar)
		
		if isinstance(strategy, str):
			assert popsize is not None
			self.strategy = self.create_strategy(strategy, popsize, params_shaper.total_params, **es_kws) # type: ignore
		else:
			self.strategy = strategy

		if es_params is None:
			self.es_params = self.strategy.default_params
		else:
			self.es_params = es_params

		self.params_shaper = params_shaper

		if eval_reps > 1:
			def _eval_fn(p: Params, k: jr.PRNGKeyArray, tp: Optional[PyTree]=None):
				"""
				"""
				fit, info = jax.vmap(task, in_axes=(None,0,None))(p, jr.split(k,eval_reps), tp)
				return jnp.mean(fit), info
			self.task = _eval_fn
		else :
			self.task = task

		if fitness_shaper is None:
			self.fitness_shaper = ex.FitnessShaper()
		else:
			self.fitness_shaper = fitness_shaper


	#-------------------------------------------------------------------

	def train_step(self, state: TrainState, key: jr.PRNGKeyArray, task_params: Optional[TaskParams]=None) -> Tuple[TrainState, Data]:
		
		ask_key, eval_key = jr.split(key, 2)
		x, state = self.strategy.ask(ask_key, state, self.es_params)
		params = self.params_shaper.reshape(x)
		fitness, eval_data = jax.vmap(self.task, in_axes=(0, None, None))(params, eval_key, task_params)
		f = self.fitness_shaper.apply(x, fitness)
		state = self.strategy.tell(x, f.astype(jnp.float32), state, self.es_params)
		
		return state, {"fitness": fitness, "data": eval_data}

	#-------------------------------------------------------------------

	def initialize(self, key: jr.PRNGKeyArray, **kwargs) -> TrainState:
		
		state = self.strategy.initialize(key, self.es_params)
		state = state.replace(**kwargs)
		return state

	#-------------------------------------------------------------------

	def create_strategy(self, name: str, popsize: int, num_dims: int, **kwargs)->ex.Strategy:
		
		ES = getattr(ex, name)
		es = ES(popsize=popsize, num_dims=num_dims, **kwargs)
		return es

