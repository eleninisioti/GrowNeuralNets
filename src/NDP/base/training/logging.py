import wandb
import jax
from jaxtyping import PyTree
from typing import Tuple, TypeAlias, Callable, Optional
from jax.experimental import host_callback as hcb
import equinox as eqx

TrainState: TypeAlias = PyTree[...]
Data: TypeAlias = PyTree[...]

class WandbLogger:

	#-------------------------------------------------------------------

	def __init__(self, metrics_fn: Callable[[TrainState, Data], Tuple[Data, Data]], 
				 chkpt_file: Optional[str]=None,
				 chkpt_freq: int=100):
		
		self.metrics_fn = metrics_fn
		self.chkpt_file = chkpt_file
		self.chkpt_freq = chkpt_freq

	#-------------------------------------------------------------------

	def __call__(self, state: TrainState, data: Data):
		
		log_data, chkpt_data = self.metrics_fn(state, data)
		self.log(log_data)
		_ = jax.lax.cond(
			(chkpt_data["epoch"] % self.chkpt_freq)==0,
			lambda d : self.save_chkpt(d),
			lambda d : None,
			chkpt_data
		)
		return log_data

	#-------------------------------------------------------------------

	def log(self, data: dict):
		hcb.id_tap(
			lambda d, *_: wandb.log(d), data
		)

	#-------------------------------------------------------------------

	def save_chkpt(self, data: dict):

		def save(data):
			assert self.chkpt_file is not None
			file = f"{self.chkpt_file}_{data['epoch']}.eqx"
			eqx.tree_serialise_leaves(file, data)

		if self.chkpt_file is not None:
			hcb.id_tap(lambda d, *_: save(d), data)

	#-------------------------------------------------------------------

	def init(self, project: str, config: dict, **kwargs):
		wandb.init(project=project, config=config, **kwargs)

	#-------------------------------------------------------------------

	def finish(self):
		wandb.finish()
