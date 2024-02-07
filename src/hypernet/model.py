import equinox as eqx
from jaxtyping import Array, Int, PyTree
from typing import NamedTuple, Optional, Union
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox.nn as nn
import evosax as ex
import equinox as eqx
from src.hypernet.base.models.gnn.layers import EMPNN, GNCA, CycleCountInjection, GAT, GraphMLP, MPNN, GraphTransformer, RMPNN, RecurrentAttentionNetwork
from typing import Mapping, NamedTuple, Tuple
import numpy as onp
import time
from flax import linen

class NDPTask(eqx.Module):
    task: PyTree
    alpha: float
    crit_size: Optional[int]
    def __init__(self, task, size_penalty, crit_size=None):
        self.task = task
        self.alpha = size_penalty
        self.crit_size = crit_size
    def __call__(self, *args, **kwargs):
        fit, data = self.task(*args, **kwargs)
        if self.crit_size is None:
            penalty = data["size"]
        else:
            penalty = (data["size"]>self.crit_size).astype(float)
        return fit - self.alpha*penalty, data


from src.hypernet.base.models.gnn.ndp import (NDP, ENDP, SimpleNDP, RecurrentNDP, off_sigmoid)
from typing import Callable



class GraphFeatures(eqx.Module):
    features: list[str]
    in_dims: Optional[int]
    rnn_iters: Optional[int]
    def __init__(self, features: list[str], in_dims: Optional[int]=None, rnn_iters: Optional[int]=None):
        assert (in_dims is not None and rnn_iters is not None) or "dynamical" not in features
        self.features=features
        self.in_dims = in_dims
        self.rnn_iters = rnn_iters
    def __call__(self, graph, key):
        h = graph.h
        A = graph.edges.A
        N = h.shape[0]
        features = jnp.zeros((N, 1))
        if "degree" in self.features:
            in_degree = A.sum(0) # N
            out_degree = A.sum(1) # N
            features = jnp.concatenate([features, in_degree[...,None], out_degree[...,None]], axis=-1)
        if "node_age" in self.features:
            assert graph.nodes.pholder is not None and hasattr(graph.nodes.pholder, "age")
            age = graph.nodes.pholder.age
            features = jnp.concatenate([features, age[:,None]], axis=-1)
        if "time" in self.features:
            assert graph.pholder is not None and hasattr(graph.pholder, "time")
            time = jnp.ones((N, 1)) * graph.pholder.time
            features = jnp.concatenate([features, time], axis=-1)
        if "cycles" in self.features:
            cycle_counter = CycleCountInjection()
            cycles = cycle_counter.count_cycles(A)
            features = jnp.concatenate([features, cycles], axis=-1)
        if "dynamical" in self.features:
            assert self.in_dims is not None and self.rnn_iters is not None
            m = graph.nodes.m
            h = jnp.zeros_like(m)
            w = graph.edges.e[...,1] * graph.edges.A
            s = jnp.zeros(h.shape[:1])
            xs = jr.normal(key, (self.rnn_iters, self.in_dims)).at[1:, :]
            s, ss = jax.lax.scan(
                lambda s, x: (rnn(xs[0], s, w),)*2, #type: ignore
                s, xs
            )
            features = jnp.concatenate([features, ss])

        h = jnp.concatenate([h, features[:, 1:]], axis=-1)
        nodes = graph.nodes._replace(h=h)
        return graph._replace(nodes=nodes)

    def n_features(self):
        mapping = {"degree": 2, "node_age": 1, "time": 1, "cycles": 5, "dynamical": self.rnn_iters}
        n = 0
        for f in self.features:
            n = n + mapping[f]
        return n

class NodeModel(eqx.Module):
    mp: PyTree
    extra_feats: Optional[GraphFeatures]
    Wout: nn.Linear
    inhibition: str
    inhibit_for: int

    def __init__(self, key, extra_features, node_features, edge_features, inhibition, inhibit_for):
        key_mp, key_mlp = jr.split(key)
        if extra_features:
            self.extra_feats = GraphFeatures(extra_features)
            n_extra_features = self.extra_feats.n_features()
        else:
            self.extra_feats = None
            n_extra_features = 0
        self.mp = GAT(node_features+n_extra_features, node_features, 2, key=key_mp, att_depth=2,
                      att_width=32, use_edges=True, edge_features=edge_features,
                      adjacency_mask=True)

        self.Wout = nn.Linear(node_features, node_features, key=key_mlp)

        self.inhibition = inhibition
        self.inhibit_for = inhibit_for

    def __call__(self, graph, key, counter):
        oh = graph.h

        if self.extra_feats is not None:
            graph = self.extra_feats(graph, None)
        graph = self.mp(graph, key)
        h = jax.vmap(self.Wout)(graph.h) # maybe remove oh from here

        @jax.jit
        def lateral_inhibition(hidden_before, hidden_after, index, inhibitions, connections):

            hidden_inhibited = jnp.copy(hidden_before)

            hidden_inhibited = jax.lax.cond(inhibitions>0,
                                      lambda x: x,
                                      lambda x: hidden_after, hidden_inhibited)
            other_inhibitions = jnp.where(connections, 1, 0)
            other_inhibitions = jnp.where(inhibitions> 0, jnp.zeros(other_inhibitions.shape), other_inhibitions)
            other_inhibitions = other_inhibitions.at[index].set(0)

            return hidden_inhibited, other_inhibitions

        if self.inhibition:
            indexes = jnp.arange(oh.shape[0])
            inhibitions = graph.nodes.inhibited_hidden
            connections = graph.edges.A

            for node in indexes:
                next_key, key =jax.random.split(key)
                h_node, inhibitions_node = lateral_inhibition(oh[node, ...], h[node, ...], node, inhibitions[node, ...], connections[node, ...] )
                #inhibitions_node = jnp.zeros(inhibitions_node.shape)
                prob_noinhib = 1/(counter+1)
                random_array = jax.random.choice(next_key, 2, shape=inhibitions.shape, p=jnp.array([prob_noinhib, 1-prob_noinhib]))
                inhibitions_node = jnp.where(random_array, 0, inhibitions_node)
                inhibitions = jnp.add(inhibitions, inhibitions_node)

                h = h.at[node,...].set(h_node)

            new_inhibitions = jnp.where(inhibitions, self.inhibit_for, graph.nodes.inhibited_hidden)
            new_inhibitions = jnp.where(new_inhibitions, new_inhibitions-1, new_inhibitions)

            graph = graph._replace(nodes=graph.nodes._replace(inhibited_hidden=new_inhibitions))

        graph = eqx.tree_at(lambda G: G.nodes.h_learned, graph, h)

        return graph


class DivModel(eqx.Module):
    mlp: nn.MLP
    stoch: bool
    intrinsic: bool
    def __init__(self, key, node_features, intrinsic,  stoch=True):

        #act = partial(off_sigmoid, off=.5, T=50.) if stoch else lambda x: x
        self.mlp = nn.MLP(node_features, 1, 32, 1, activation=linen.relu, final_activation=jax.nn.tanh, key=key)
        self.stoch = stoch
        self.intrinsic = intrinsic

    def __call__(self, graph, key):
        if self.intrinsic:
            hidden_state = jnp.concatenate([graph.h, graph.nodes.h_intrinsic], axis=1)
        else:
            hidden_state = graph.h
        d = jax.vmap(self.mlp)(hidden_state)
        if self.stoch:
            d = (jr.uniform(key, d.shape)<d).astype(float)
        else:
            d = (d>0.).astype(float)
        return d

class EdgeModel(eqx.Module):
    mlp: nn.MLP

    def __call__(self, e, pre, post):
        temp =jnp.concatenate([pre, post], axis=-1)
        return self.mlp(jnp.concatenate([pre, post], axis=-1))



from typing import NamedTuple
from src.NDP.base.models.gnn.base import Graph, Node, Edge

class NodeInfo(NamedTuple):
    age: Int[Array, "N"]

class GraphInfo(NamedTuple):
    time: int

class CustomNDP(ENDP):

    def __call__(self, graph, key, counter):
        graph = super().__call__(graph, key, counter)
        assert isinstance(graph.pholder, GraphInfo)
        assert isinstance(graph.nodes.pholder, NodeInfo)
        time = graph.pholder.time
        nodes_age = graph.nodes.pholder.age
        graph = eqx.tree_at(
              lambda g: [g.nodes.pholder.age, g.pholder.time],
              graph,
              [nodes_age+graph.nodes.m, time+1]
        )
        return graph

    def initialize(self, key):
        factor = 1
        cpp_h = jnp.arange(self.max_nodes)
        #cpp_h = jnp.concatenate([jnp.arange(self.init_nodes),jnp.arange(self.max_nodes-self.init_nodes)])

        #cpp_h = jax.random.choice(key, cpp_h.size ,shape=(self.max_nodes,), replace=False)
        #cpp_h = cpp_h / (jnp.linalg.norm(cpp_h, axis=-1, keepdims=True) + 1e-8)
        #cpp_h = jax.random.shuffle(key, cpp_h)
        cpp_h = jax.nn.one_hot(cpp_h, num_classes=self.max_nodes*factor)
        #cpp_h = cpp_h.at[self.init_nodes:,...].set(0)
        #cpp_h = cpp_h[..., :self.node_features_intrinsic]


        nodes = Node(
            inhibited_edge=jnp.zeros(self.max_nodes),
            inhibited_node=jnp.zeros(self.max_nodes),
            inhibited_hidden=jnp.zeros(self.max_nodes),
            h_learned=jax.random.normal(key,(self.max_nodes, self.node_features)),
            h_intrinsic=cpp_h,
            #m=jnp.zeros((self.max_nodes,)).at[:self.init_nodes].set(1.),
            m=jnp.ones((self.max_nodes,)),
            #m=jnp.zeros((self.max_nodes,)),
            pholder=NodeInfo(age=jnp.zeros((self.max_nodes,))))
        edges = Edge(
            A=jnp.zeros((self.max_nodes, self.max_nodes)),
            e=jr.normal(key, (self.max_nodes, self.max_nodes, self.edge_features))*onp.sqrt(0.01))
            #e = jnp.zeros((self.max_nodes, self.max_nodes, self.edge_features)) )


        return Graph(nodes=nodes, edges=edges, pholder=GraphInfo(time=0))



def rnn(x, h, w):
    d = x.shape[-1]
    h = h.at[:d].set(x)
    h = jnn.tanh(h@w)
    return h

from typing import NamedTuple
class PolicyState(NamedTuple):
    w: jax.Array
    m: jax.Array
    h: jax.Array
    hidden: jax.Array
    b: Optional[jax.Array] = None

    a: Optional[jax.Array] = None


class NDPToRNN(eqx.Module):
    #-------------------------------------------------------------------
    ndp: NDP
    dev_steps: int
    action_dims: int
    obs_dims: int
    action_type: str
    input_mode: str
    bias: bool
    policy_iters: int
    discrete_action: bool
    is_recurrent: bool
    h0: Optional[jax.Array]
    trainable_init: bool
    add_bias: False

    #-------------------------------------------------------------------
    def __init__(self, ndp: NDP, dev_steps: int, action_dims: int, obs_dims:int, action_type: str="d", trainable_init: bool=True, *, key: jax.Array):
        self.ndp = ndp
        self.dev_steps = dev_steps
        self.action_dims = action_dims
        self.action_type = action_type
        self.h0 = jr.normal(key, (ndp.max_nodes, ndp.node_features)) if trainable_init else None
        self.trainable_init = trainable_init
        self.obs_dims = obs_dims
        self.input_mode = "set"
        self.bias = False
        self.policy_iters = 3
        self.discrete_action = False
        self.is_recurrent = True
        self.add_bias = False


    #-------------------------------------------------------------------
    def __call__(self, obs: jax.Array, state: PolicyState, key: jax.Array) -> Tuple[jax.Array, PolicyState]:

        def rnn_step(h):
            return jnn.tanh(h @ state.w)

        if self.input_mode == "set":
            h = state.h.at[:self.obs_dims].set(obs)
        elif self.input_mode == "add":
            h = state.h.at[:self.obs_dims].add(obs)
        else:
            raise NotImplementedError(f"input mode {self.input_mode} is not a valid mode, use either 'add' or 'set'")
        if self.add_bias:
            h = h.at[obs.shape[0]].set(1.)
        h = jax.lax.fori_loop(0, self.policy_iters, lambda _, h: rnn_step(h), h)
        if state.a is None:
            a = h[-self.action_dims:]
        else:
            a = jnp.take(h, state.a)
        if self.discrete_action:
            a = jnp.argmax(a)
        if self.is_recurrent:
            state = state._replace(h=h)
        return a, state
    #-------------------------------------------------------------------

    def initialize(self, key: jax.Array, eval=False)->PolicyState:
        key_init, key_rollout = jr.split(key)
        G = self.init_graph(key_init)
        #h_updated_before_rollout = G.nodes.h
        G, all_graphs = self.ndp.rollout(G, key_rollout, self.dev_steps)


        def get_policy_state(graph):
            m = graph.nodes.m
            h = jnp.zeros_like(m)
            w = graph.edges.e[..., 0] * graph.edges.A

            #print(graph.edges.e[..., 1] )
            #print(graph.edges.e[..., 1] )


            h_updated = jnp.concatenate([graph.nodes.h_intrinsic, graph.nodes.h_learned], axis=1)
            return PolicyState(w=w,m=m,h=h,hidden=h_updated)



        #for graph_idx in range(self.dev_steps):
        #    graph = jax.tree_util.tree_map(lambda x: x[graph_idx, ...] if isinstance(x, jax.numpy.ndarray) else x, all_graphs)
        #    get_policy_state(graph)

        policy_states = jax.vmap(get_policy_state, in_axes=(0))(all_graphs)

        m = G.nodes.m
        h = jnp.zeros_like(m)
        w = G.edges.e[...,0] * G.edges.A


        h_updated = jnp.concatenate([G.nodes.h_intrinsic, G.nodes.h_learned], axis=1)
        return PolicyState(w=w, m=m, h=h, hidden=h_updated), policy_states
    #-------------------------------------------------------------------
    def init_graph(self, key: jax.Array):
        G = self.ndp.initialize(key)
        #self.trainable_init = False
        if self.trainable_init:
            #nodes = G.nodes._replace(h=G.h.at[:self.NDP_scripts.init_nodes].set(self.h0[0,...]))
            #nodes = G.nodes._replace(h=G.h.at[:self.NDP_scripts.init_nodes,:].set(self.h0[:self.NDP_scripts.init_nodes,:]))
            nodes = G.nodes


        else:
            nodes = G.nodes
        return G._replace(nodes=nodes)


def make_model(config, key):
    """ Creates the NDP model.
    """

    key, key_model = jr.split(key)
    key_node, key_edge, key_div = jr.split(key_model, 3)
    action_type = "c"
    edge_features = 1
    extra_features = []
    action_size = config["env_config"]["action_size"]
    input_size = config["env_config"]["obs_size"]
    max_nodes = config["model_config"]["max_hidden_neurons"] + action_size + input_size
    node_features_learned = 8
    init_nodes = input_size + action_size


    inhibit_hidden = False
    inhibit_mitosis = False
    inhibit_weights = False
    config["model_config"]["intrinsic"] = True

    if config["model_config"]["intrinsic"]:
        node_features_intrinsic = max_nodes
    else:
        node_features_intrinsic = 1
    # node_fn learns hidden states
    node_fn = NodeModel(key_node,
                        extra_features,
                        node_features_learned,
                        edge_features,
                        inhibition=inhibit_hidden,
                        inhibit_for=config["model_config"]["inhibit_for"])

    # edge_fn learns weight values
    if config["model_config"]["intrinsic"]:
        node_features = node_features_intrinsic
    else:
        node_features = node_features_learned

    edge_fn = nn.MLP(2 * node_features,
                     edge_features,
                     16, 2, key=key_edge, activation=linen.leaky_relu,
                     final_activation=lambda x: x, use_bias=False, use_final_bias=False)

    edge_fn = EdgeModel(mlp=edge_fn)

    # div_fn learns neuron mitosis
    div_fn = DivModel(key_div, node_features, stoch=False, intrinsic=config["model_config"]["intrinsic"])

    ndp = CustomNDP(max_nodes,
                    init_nodes,
                    node_features_learned,
                    edge_features,
                    node_fn,
                    edge_fn,
                    div_fn,
                    node_features_intrinsic=node_features_intrinsic,
                    inhibit_mitosis=inhibit_mitosis,
                    inhibit_weights=inhibit_weights,
                    inhibit_for=config["model_config"]["inhibit_for"],
                    intrinsic=config["model_config"]["intrinsic"])  # type: ignore

    model = NDPToRNN(ndp, config["model_config"]["dev_steps"], action_size, input_size, action_type=action_type, key=key_model)

    return model
