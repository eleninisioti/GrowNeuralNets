import wandb
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from typing import Tuple, TypeAlias, Callable, Optional
from jax.experimental import host_callback as hcb
import equinox as eqx
import os
import pickle
TrainState: TypeAlias = PyTree[...]
Data: TypeAlias = PyTree[...]

class Logger:

	#-------------------------------------------------------------------

	def __init__(
		self, 
		wandb_log: bool,
		metrics_fn: Callable[[TrainState, Data], Tuple[Data, Data, int]], 
		ckpt_dir: Optional[str]=None,
		ckpt_freq: int=100,
		dev_steps: int=None,
		verbose: bool=False):

		self.wandb_log = wandb_log
		self.metrics_fn = metrics_fn
		self.ckpt_dir = ckpt_dir
		self.ckpt_freq = ckpt_freq
		self.epoch = [0]
		self.verbose = verbose
		self.dev_steps = dev_steps

	#-------------------------------------------------------------------

	def log(self, state: TrainState, data: Data):
		
		log_data, ckpt_data, epoch, interm_data, best_indiv = self.metrics_fn(state, data)
		if self.wandb_log:
			self._log(log_data)

		self.save_best_model(ckpt_data, epoch)

		for dev_step in range(self.dev_steps):
			current_dev =  jax.tree_map(lambda x: x[best_indiv, 0, dev_step,...], interm_data)
			self.save_chkpt(current_dev, epoch, dev_step)
		return log_data

	def save_best_model(self, data: dict, epoch: int):

		def save(d):
			data, epoch = d
			assert self.ckpt_dir is not None
			save_dir= self.ckpt_dir + "/best_model"
			if not os.path.exists(save_dir):
				os.makedirs(save_dir, exist_ok=True)

			file = save_dir + "/ckpt.eqx"
			if self.verbose:
				print("saving data at: ", file)
			eqx.tree_serialise_leaves(file, data)

		def tap_save(data, epoch):
			hcb.id_tap(lambda d, *_: save(d), (data, epoch))
			return None

		if self.ckpt_dir is not None:
			jax.lax.cond(
				(jnp.mod(epoch, self.ckpt_freq)) == 0,
				lambda data: tap_save(data, epoch),
				lambda data: None,
				data
			)

	#-------------------------------------------------------------------

	def _log(self, data: dict):
		hcb.id_tap(
			lambda d, *_: wandb.log(d), data
		)

	#-------------------------------------------------------------------

	def save_chkpt(self, data: dict, epoch: int, dev_step: int):

		def save(d):
			data, epoch, dev_step =d
			assert self.ckpt_dir is not None
			save_dir= self.ckpt_dir + "/all_info/gen_" + str(epoch)
			if not os.path.exists(save_dir):
				os.makedirs(save_dir, exist_ok=True)

			file = save_dir + "/dev_" + str(dev_step) + ".pkl"

			if self.verbose:
				print("saving data at: ", file)
			with open(file, "wb") as f:
				pickle.dump( data,f)
			#eqx.tree_serialise_leaves(file, data)

		def tap_save(data, epoch, dev_step):

			hcb.id_tap(lambda d, *_: save(d), (data, epoch, dev_step))
			return None

		if self.ckpt_dir is not None:
			jax.lax.cond(
				(jnp.mod(epoch, self.ckpt_freq))==0,
				lambda data : tap_save(data, epoch, dev_step),
				lambda data : None,
				data
			)

	#-------------------------------------------------------------------

	def wandb_init(self, project: str, config: dict, **kwargs):
		if self.wandb_log:
			wandb.init(project=project, config=config, **kwargs)

	#-------------------------------------------------------------------

	def wandb_finish(self, *args, **kwargs):
		if self.wandb_log:
			wandb.finish(*args, **kwargs)
