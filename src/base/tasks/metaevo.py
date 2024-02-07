from functools import partial
from src.training.evolution import EvosaxTrainer, Params, TrainState, Data
from src.tasks.base import BaseTask
import evosax as ex

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx

from typing import Union, Callable, Optional, Dict, Any, Tuple
from jaxtyping import Float, PyTree


def avg_fitness(state, data):
	return data["eval_data"]["fitness"].mean()

def end_avg_fitness(state, data):
	return data["eval_data"]["fitness"][-1].mean()

def best_fitness(state: ex.EvoState, data):
	return data["states"].best_fitness[-1]

def min_min(state: ex.EvoState, data):
	return data["states"].best_fitness[-1]

Params = PyTree[...]
TaskParams = PyTree[...]

class MetaEvolutionTask(BaseTask):
	
	"""
	"""
	#-------------------------------------------------------------------
	trainer: EvosaxTrainer
	fitness_fn: Callable[[TrainState, Data], Float]
	task_sampler: Callable[[jr.PRNGKeyArray], TaskParams]
	#-------------------------------------------------------------------

	def __init__(
		self, 
		train_steps: int, 
		task: Callable[[Params, jr.PRNGKeyArray], Float], 
		params_shaper: ex.ParameterReshaper, 
		task_sampler: Callable[[jr.PRNGKeyArray], TaskParams],
		strategy: Union[ex.Strategy, str]="DES", 
		fitness_fn: Callable[[TrainState, Data], Float]=min_min,
		fitness_shaper: ex.FitnessShaper=ex.FitnessShaper(),
		popsize: Optional[int] = 64, 
		es_kws: Optional[Dict[str, Any]] = {}, 
		es_params: Optional[ex.EvoParams] = None, 
		eval_reps: int = 1):

		self.trainer = EvosaxTrainer(train_steps, strategy, task, 
									 params_shaper, popsize, es_kws=es_kws, 
									 es_params=es_params, eval_reps=eval_reps,
									 fitness_shaper=fitness_shaper)
		self.fitness_fn = fitness_fn
		self.task_sampler = task_sampler

	#-------------------------------------------------------------------

	def __call__(self, outer_params: Params, key: jr.PRNGKeyArray, task_params: TaskParams = None) -> Tuple[Float, Data]:
		
		key_task, key_train = jr.split(key)
		task_params = self.task_sampler(key_task)
		es_state, data = self.init_and_train(outer_params, key_train, task_params)
		fit = self.fitness_fn(es_state, data) #type: ignore
		return fit, data

	#-------------------------------------------------------------------

	def train_step(self, outer_params: Params, state: TrainState, key: jr.PRNGKeyArray, task_params: Optional[TaskParams]=None) -> Tuple[TrainState, Data]:
		
		ask_key, eval_key = jr.split(key, 2)
		x, state = self.trainer.strategy.ask(ask_key, state, self.trainer.es_params)
		dna = self.trainer.params_shaper.reshape(x)

		@partial(jax.vmap, in_axes=(0,None,None))
		def _eval(dna, key, task_params):
			params = eqx.combine(outer_params, dna)
			fitness, eval_data = self.trainer.task(params, key, task_params)
			return fitness, eval_data
		
		fitness, eval_data = _eval(dna, eval_key, task_params)
		fitness_sh = self.trainer.fitness_shaper.apply(fitness)
		state = self.trainer.strategy.tell(x, fitness_sh, state, self.trainer.es_params)
		
		return state, {"fitness": fitness, "data": eval_data}

	#-------------------------------------------------------------------

	def train(self, outer_params, state, key, task_params):
		def _step(c, x):
			s, k = c
			k, k_ = jr.split(k)
			s, data = self.train_step(outer_params, s, k_, task_params)
			return [s, k], {"states": s, "eval_data": data}

		[state, key], data = jax.lax.scan(_step, [state, key], None, self.trainer.train_steps)
		return state, data

	#-------------------------------------------------------------------

	def init_and_train(self, outer_params: Params, key: jr.PRNGKeyArray, task_params: Optional[TaskParams]=None)->Tuple[TrainState, Data]:
		init_key, train_key = jr.split(key)
		state = self.trainer.initialize(init_key)
		return self.train(outer_params, state, train_key, task_params)




class MetaDevoTask(BaseTask):
	
	"""
	"""
	#-------------------------------------------------------------------
	trainer: EvosaxTrainer
	fitness_fn: Callable[[TrainState, Data], Float]
	task_sampler: Callable[[jr.PRNGKeyArray], TaskParams]
	statics: PyTree
	#-------------------------------------------------------------------

	def __init__(
		self, 
		statics: PyTree,
		train_steps: int, 
		task: Callable[[Params, jr.PRNGKeyArray], Float], 
		params_shaper: ex.ParameterReshaper, 
		task_sampler: Callable[[jr.PRNGKeyArray], TaskParams],
		strategy: Union[ex.Strategy, str]="DES", 
		fitness_fn: Callable[[TrainState, Data], Float]=min_min,
		fitness_shaper: ex.FitnessShaper=ex.FitnessShaper(),
		popsize: Optional[int] = 64, 
		es_kws: Optional[Dict[str, Any]] = {}, 
		es_params: Optional[ex.EvoParams] = None, 
		eval_reps: int = 1):

		self.trainer = EvosaxTrainer(train_steps, strategy, task, 
									 params_shaper, popsize, es_kws=es_kws, 
									 es_params=es_params, eval_reps=eval_reps,
									 fitness_shaper=fitness_shaper)
		self.statics = statics
		self.fitness_fn = fitness_fn
		self.task_sampler = task_sampler

	#-------------------------------------------------------------------

	def __call__(self, outer_params: Params, key: jr.PRNGKeyArray, task_params: TaskParams = None) -> Tuple[Float, Data]:
		
		key_task, key_train = jr.split(key)
		task_params = self.task_sampler(key_task)
		es_state, data = self.init_and_train(outer_params, key_train, task_params)
		fit = self.fitness_fn(es_state, data) #type: ignore
		return fit, data

	#-------------------------------------------------------------------

	def train_step(self, outer_params: Params, state: TrainState, key: jr.PRNGKeyArray, task_params: Optional[TaskParams]=None) -> Tuple[TrainState, Data]:
		
		ask_key, eval_key, dev_key = jr.split(key, 3)
		# Make developemntal model
		outer_model = eqx.combine(outer_params, self.statics)
		# Unroll development
		outer_res = outer_model(dev_key)
		# Get inner params
		x, state = self.trainer.strategy.ask(ask_key, state, self.trainer.es_params)
		inner_params = self.trainer.params_shaper.reshape(x)

		@partial(jax.vmap, in_axes=(0,None,None))
		def _eval(inner_params, key, task_params):
			params = eqx.combine(inner_params, outer_res)
			fitness, eval_data = self.trainer.task(params, key, task_params)
			return fitness, eval_data
		
		fitness, eval_data = _eval(inner_params, eval_key, task_params)
		fitness_sh = self.trainer.fitness_shaper.apply(x, fitness)
		state = self.trainer.strategy.tell(x, fitness_sh, state, self.trainer.es_params)
		
		return state, {"fitness": fitness, "data": eval_data}

	#-------------------------------------------------------------------

	def train(self, outer_params, state, key, task_params):
		def _step(c, x):
			s, k = c
			k, k_ = jr.split(k)
			s, data = self.train_step(outer_params, s, k_, task_params)
			return [s, k], {"states": s, "eval_data": data}

		[state, key], data = jax.lax.scan(_step, [state, key], None, self.trainer.train_steps)
		return state, data

	#-------------------------------------------------------------------

	def init_and_train(self, outer_params: Params, key: jr.PRNGKeyArray, task_params: Optional[TaskParams]=None)->Tuple[TrainState, Data]:
		init_key, train_key = jr.split(key)
		state = self.trainer.initialize(init_key)
		return self.train(outer_params, state, train_key, task_params)
























