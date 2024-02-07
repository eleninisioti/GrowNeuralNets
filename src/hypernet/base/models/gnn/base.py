import jax
import equinox as eqx
import jax.random as jr
from typing import Optional, Tuple, NamedTuple
from jaxtyping import Float, Array, PyTree, Int, Bool

class Node(NamedTuple):
	#-------------------------------------------------------------------
	h_intrinsic: Float[Array, "N Dn"]
	h_learned: Float[Array, "N Dn"]
	m: Optional[Float[Array, "N"]]=None
	p: Optional[Float[Array, "N X"]]=None
	pholder: Optional[PyTree]=None
	inhibited_node: Optional[Bool]=False
	inhibited_edge: Optional[Bool]=False
	inhibited_hidden: Optional[Bool] = False


#-------------------------------------------------------------------

class Edge(NamedTuple):
	#-------------------------------------------------------------------
	A: Optional[Float[Array, "..."]]=None
	senders: Optional[Int[Array, "E"]]=None
	receivers: Optional[Int[Array, "E"]]=None
	e: Optional[Float[Array, "E De"]]=None
	m: Optional[Float[Array, "E"]]=None
	pholder: Optional[PyTree]=None
	#-------------------------------------------------------------------

class Graph(NamedTuple):
	#-------------------------------------------------------------------
	nodes: Node
	edges: Edge
	pholder: Optional[PyTree]=None
	#-------------------------------------------------------------------
	@property
	def h(self):
		return self.nodes.h_learned
	#-------------------------------------------------------------------
	@property
	def A(self):
		return self.edges.A
	#-------------------------------------------------------------------
	@property
	def N(self):
		return self.nodes.h.shape[0]


class GraphModule(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	#-------------------------------------------------------------------

	def __call__(self, graph: Graph, key: jax.Array)->Graph:
		if graph.edges.receivers is not None and graph.edges.senders is not None:
			return self.apply_list(graph, key)
		elif graph.edges.A is not None:
			return self.apply_adj(graph, key)
		else :
			raise ValueError(f"No data provided for edges not an edge encoding method, use either adj or list")

	#-------------------------------------------------------------------

	def apply_list(self, graph: Graph, key: jax.Array)->Graph:
		raise NotImplementedError(f"{self.__class__} is not implemented for graph with adjacency list encoding")

	#-------------------------------------------------------------------

	def apply_adj(self, graph: Graph, key: jax.Array)->Graph:
		raise NotImplementedError(f"{self.__class__} is not implemented for graph with adjacency matrix encoding")

	#-------------------------------------------------------------------


class IterativeGraphModule(GraphModule):

	#-------------------------------------------------------------------

	def rollout(self, graph: Graph, key: jax.Array, steps: int)->Tuple[Graph, Graph]:

		def step(carry, x):
			graph, key = carry
			key, subkey = jr.split(key)
			graph = self.__call__(graph, subkey)
			return [graph, key], graph

		[graph, _], graphs = jax.lax.scan(step, [graph, key], None, steps)
		return graph, graphs

	#-------------------------------------------------------------------

		