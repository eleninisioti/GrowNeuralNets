from copy import Error
from typing import Optional
from src.gnn.base import Graph, Node, Edge
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import numpy as np
import os
from src.gnn.utils import from_graphml
import networkx as nx
from torch.utils import data
import numpy as np
import equinox as eqx

def load_graphml(path, nX, nE, max_nodes=None):
	nxG = nx.read_graphml(path)
	Ndata = len(nxG.nodes)
	N =  Ndata if max_nodes is None else max_nodes
	if N < Ndata:
		raise ValueError("data has more nodes than max_nodes")
	if max_nodes is None:
		X = jnp.zeros((N, nX)).at[:, 1].set(1.)
		E = jnp.array(nx.adjacency_matrix(nxG).todense())
		E = jnn.one_hot(E, nE)
	else :
		X = jnp.zeros((N, nX)).at[:N, 1].set(1.)
		E = jnp.array(nx.adjacency_matrix(nxG).todense())
		E = jnp.pad(E, ((0, N-Ndata), (0, N-Ndata)))
		E = jnn.one_hot(E, nE)

	return (X, E), N

def load_processed_data(path, nX, nE, N):
	like = (jnp.zeros((N, nX)), jnp.zeros((N,N,nE)))
	return eqx.tree_deserialise_leaves(path, like)

class BrainGraphDataset(data.Dataset):

	#-------------------------------------------------------------------

	def __init__(self, path: str, nx: int, ne: int, max_nodes: Optional[int]=None,
				 force_pre_processing: bool=False, subset: Optional[float|int]=None, 
				 key: Optional[jax.Array]=None):
		prpath = path + "_pr"
		if max_nodes is not None:
			self.path = prpath + str(max_nodes)
		else:
			self.path = prpath

		if subset is None:
			og_files = list(os.listdir(path))
		else:
			assert key is not None
			og_files = list(os.listdir(path))
			N = len(og_files)
			n = int(subset*N) if isinstance(subset, float) else min([subset, N])
			ids = jr.choice(key, jnp.arange(N), (n,), replace=False)
			og_files = [og_files[i] for i in ids]

		self.files = [f.replace(".graphml", ".eqx") for f in og_files]
		self.nx = nx
		self.ne = ne
		self.max_nodes = max_nodes
		self.pre_process_data(path, force=force_pre_processing)

	#-------------------------------------------------------------------

	def __len__(self):
		return len(self.files)

	#-------------------------------------------------------------------

	def pre_process_data(self, path, force=False):
		
		print("pre processing the data")
		if force and os.path.isdir(self.path):
			os.removedirs(self.path)
		
		if os.path.isdir(self.path):
			print("data has already been processed")
			if self.max_nodes is None:
				og_files = list(os.listdir(path))
				ex_filepath = f"{path}/{og_files[0]}"
				_, N = load_graphml(ex_filepath, self.nx, self.ne)
				self.max_nodes = N
			return
		
		else:
			os.makedirs(self.path)
			G = None
			og_files = list(os.listdir(path))
			files = [f.replace(".graphml", ".eqx") for f in og_files]
			for og_file, pr_file in zip(og_files, files):
				og_filepath = f"{path}/{og_file}"
				pr_filepath = f"{self.path}/{pr_file}"
				G, N = load_graphml(og_filepath, nX=self.nx,nE=self.ne,max_nodes=self.max_nodes)
				if self.max_nodes is None:
					self.max_nodes = N
				eqx.tree_serialise_leaves(pr_filepath, G)
			print("pre processing finsished")
			return G

	#-------------------------------------------------------------------

	def __getitem__(self, index):
		filepath = f"{self.path}/{self.files[index]}"
		G = load_processed_data(filepath, self.nx, self.ne, self.max_nodes)
		return G

	#-------------------------------------------------------------------

def stack_trees(batch):
	batch = jax.tree_map(jnp.array, batch)
	batch = jax.tree_map(lambda *x: jnp.stack(x), *batch)
	return batch

class JaxLoader(data.DataLoader):

	def __init__(self, dataset, batch_size=1,
				 shuffle=False, sampler=None,
				 batch_sampler=None, num_workers=0,
				 pin_memory=False, drop_last=False,
				 timeout=0, worker_init_fn=None):
		assert batch_size <= len(dataset)
		super(self.__class__, self).__init__(dataset,
											 batch_size=batch_size,
											 shuffle=shuffle,
											 sampler=sampler,
											 batch_sampler=batch_sampler,
											 num_workers=num_workers,
											 collate_fn=stack_trees, #type:ignore
											 pin_memory=pin_memory,
											 drop_last=drop_last,
											 timeout=timeout,
											 worker_init_fn=worker_init_fn)



if __name__ == '__main__':
	import matplotlib.pyplot as plt
	path = "datasets/braingraph/86"
	dataset = BrainGraphDataset(path, 2, 2, max_nodes=None, force_pre_processing=False)
	X, E = dataset[0]
	print(X.shape, E.shape)