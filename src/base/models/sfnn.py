from collections import namedtuple
from typing import Callable, NamedTuple, Optional, Tuple, Union, TypeAlias

from jaxtyping import Float, Int, Array, PyTree
from src.gnn.layers import GAT
from src.nn.layers import RNN, MGU

from src.training.evolution import EvosaxTrainer
from src.training.logging import Logger

from src.tasks.rl import GymnaxTask, BraxTask

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn
import evosax as ex

RecurrentNeuralNetwork: TypeAlias = Union[nn.GRUCell, MGU, RNN]

class Network(NamedTuple):
	h: Float[Array, "N dh"] # Nodes hidden states
	types: Int[Array, "N"]   # Node types
	e: Float[Array, "N N de"]# Edges hidden states
	A: Float[Array, "N N"]   # Adjacency matrix
	r: Float   				 # reward

class SFNN(eqx.Module):
	"""
	Structurally Flexible Neural Network
	"""
	#-------------------------------------------------------------------
	# Parameters:
	node_cells: RecurrentNeuralNetwork
	node_output: nn.Sequential
	edge_cells: RecurrentNeuralNetwork
	# Statics:
	dh: int
	de: int
	n_types: int
	n_nodes: int
	action_dims: int
	#-------------------------------------------------------------------

	def __init__(self, hidden_dims: int, msg_dims: int, n_types: int, n_nodes: int, 
				 action_dims: int, cell_type: str="gru", activation: Callable=jnn.relu, *, key: jax.Array):


		self.dh = hidden_dims
		self.de = msg_dims
		self.n_types = n_types
		self.n_nodes = n_nodes
		self.action_dims = action_dims

		key_ncells, key_nout, key_ecells = jr.split(key, 3)
		
		def init_node_cell(key):
			if cell_type == "gru":
				return nn.GRUCell(msg_dims, hidden_dims, key=key)
			elif cell_type == "mgu":
				return MGU(msg_dims, hidden_dims, key=key)
			elif cell_type == "rnn":
				return RNN(hidden_dims, msg_dims, key=key)
			else : 
				raise ValueError(f"{cell_type} is not a known or managed cell type")

		self.node_cells = jax.vmap(init_node_cell)(jr.split(key_ncells, n_types))
		self.node_output = nn.Sequential(
			[
				nn.Linear(hidden_dims, msg_dims, key=key_nout),
				nn.Lambda(activation)
			]
		)

		def init_edge_cell(key):
			in_dims = 2*msg_dims+1
			out_dims=msg_dims
			if cell_type == "gru":
				return nn.GRUCell(in_dims, out_dims, key=key)
			elif cell_type == "mgu":
				return MGU(in_dims, out_dims, key=key)
			elif cell_type == "rnn":
				return RNN(in_dims, out_dims, key=key)
			else : 
				raise ValueError(f"{cell_type} is not a known or managed cell type")

		self.edge_cells = jax.vmap(init_edge_cell)(jr.split(key_ecells, n_types))

	#-------------------------------------------------------------------

	def __call__(self, obs: jax.Array, net: Network, key: Optional[jax.Array]=None) -> Tuple[Int, Network]:
		"""
		TODO: 
			add obs as input to the network
		"""

		h = net.h 	  # N x H
		A = net.A	  # N x N
		e = net.e 	  # N x N x M
		N = h.shape[0]
		node_types = net.types # N: int
		edge_types = jnp.repeat(node_types[:,None], N, axis=1) # N x N: int

		# 1. compute and aggregate signals
		y = jax.vmap(self.node_output)(h) 		  # N x M
		m = y[:, None, :] * (e * A[...,None])  	  # N x N x M
		x = m.sum(1) 		  			 		  # N x M

		x = x.at[:obs.shape[0], 0].set(obs) # set first element of first *obs_size* neurons to represent the obs

		# 2. Update node states
		h = jax.vmap(self._apply_node_cell)(x, h, node_types) # N x H

		# 3. Update edges states
		yiyjr = jnp.concatenate(	# N x N x 2M+1
			[
				jnp.repeat(y[:,None], N, axis=1),
			 	jnp.repeat(y[None,:], N, axis=0),
			 	jnp.ones((N,N,1)) * net.r
		 	],
			 axis=-1
		)
		e = jax.vmap(jax.vmap(self._apply_edge_cell))(yiyjr, e, edge_types) # N x N x M

		# Get action
		a = y[-self.action_dims:, 0]
		a = jnp.argmax(a)

		return a, net._replace(h=h, e=e)

	#-------------------------------------------------------------------

	def _apply_node_cell(self, x: jax.Array, h: jax.Array, typ: Int):
		cell = jax.tree_map(lambda x: x[typ] if eqx.is_array(x) else x, self.node_cells)
		return cell(x, h)

	def _apply_edge_cell(self, x: jax.Array, e: jax.Array, typ: Int):
		cell = jax.tree_map(lambda x: x[typ] if eqx.is_array(x) else x, self.edge_cells)
		return cell(x, e)

	#-------------------------------------------------------------------

	def initialize(self, key: jax.Array)->Network:
		key_types, key_A = jr.split(key)
		h = jnp.zeros((self.n_nodes, self.dh))
		e = jnp.zeros((self.n_nodes, self.n_nodes, self.de))
		types = jr.randint(key_types, (self.n_nodes,), 0, self.n_types)
		A = jr.randint(key_A, (self.n_nodes, self.n_nodes), 0, 2)
		return Network(h=h, e=e, types=types, A=A, r=0.)

#=======================================================================

class Config(NamedTuple):
	seed: int=1
	# --- Model ---
	N: int = 32 # number of nodes
	dh: int = 8 # node states features
	de: int = 3 # edge state features
	n_types: int = 4 # Number of distinct node types
	# --- Env ---
	env_name: str = "CartPole-v1"
	env_lib: str = "gymnax"
	# ---Optimizer ---
	strategy: str = "CMA_ES"
	popsize: int = 256
	generations: int = 1024
	wandb_log: bool=False
	eval_reps: int=1 #Number of evaluations ftiness is averaged over (monte carlo samplings)

default_config = Config()

#=======================================================================

if __name__ == '__main__':


	# --- Setup Config ---
	config = default_config
	# --- create Model ---
	model = SFNN(hidden_dims=config.dh,
				 msg_dims=config.de,
				 n_types=config.n_types,
				 n_nodes=config.N,
				 action_dims=2,
				 key=jr.key(1))

	params, statics = eqx.partition(model, eqx.is_array)

	params_shaper = ex.ParameterReshaper(params)

	# --- create Task ---
	def data_fn(data: dict):
		"""function computing some data about what the model is doing"""
		return {}

	if config.env_lib == "brax":
		task = BraxTask(statics, env=config.env_name, max_steps=1_000, data_fn=data_fn)
	else:
		task = GymnaxTask(statics, env=config.env_name, data_fn=data_fn)

	# Note:
	# Task just have to be a callable taking parameters, ramdom generation key and 
	# optional data and outputting fitness and some optional data
	Task: TypeAlias = Callable[[PyTree, jax.Array, Optional[PyTree]], 
							    Tuple[Float, PyTree]]

	# --- Create trainer ---
	def metrics_fn(state, data):
		"""function computing training metrics for logging"""
		y = {}
		y["best"] = - state.best_fitness
		y["gen_best"] = data["fitness"].max()
		y["gen_mean"] = data["fitness"].mean()
		y["gen_worse"] = data["fitness"].min()
		y["var"] = jnp.var(data["fitness"])
		return y, state.best_member, state.gen_counter

	logger = Logger(wandb_log=config.wandb_log, metrics_fn=metrics_fn)
	fitness_shaper = ex.FitnessShaper(maximize=True)
	trainer = EvosaxTrainer(config.generations,
							config.strategy,
							task,
							params_shaper,
							config.popsize,
							fitness_shaper,
							eval_reps=config.eval_reps)

	trainer.init_and_train_(jr.key(config.seed))










