import os
import sys
sys.path.append(os.getcwd())
import yaml
import equinox as eqx
import random
import pickle
import jax
import jax.random as jr
import evosax
from src.NDP.model import make_model
from brax.io import html
from src.base.tasks.rl import BraxTask
from src.toolkit.viz_NDP import viz_hidden, viz_network, viz_weights,  viz_network_colored, viz_graph_growth, viz_hidden_growth
from src.toolkit.analyze_NDP import analyze_graph, analyze_hidden


def load_best_model(ckpt_file, config):
    model = make_model(config, jr.key(1))
    p_like, statics = eqx.partition(model, eqx.is_array)
    pshaper = evosax.ParameterReshaper(p_like)
    pflat_like = pshaper.flatten_single(p_like)

    params_flat = eqx.tree_deserialise_leaves(ckpt_file, pflat_like)
    params = pshaper.reshape_single(params_flat)
    model = eqx.combine(params, statics)
    return model, statics

def load_NDP_info(data_dir):
    with open(data_dir, "rb") as f:
        model_data = pickle.load(f)

    model_data = {"weights": model_data.w,
                  "hidden": model_data.hidden,
                  "mask": model_data.m}
    return model_data

# --------------------------------------------

def viz_trajectory(project_dir, trial, config):
    model, statics = load_best_model(project_dir + "/trial_" + str(trial) + "/data/best_model/ckpt.eqx", config)
    env_config = config["env_config"]

    def data_fn(data: dict):
        return {}
    env = BraxTask(statics,
                   env_config["env_name"],
                   env_config["max_steps"],
                   env_config["backend"],
                   data_fn=data_fn,
                   env_kwargs={})
    import jax.numpy as jnp
    import numpy as onp
    total_rewards = []
    for eval_trial in range(10):
        state, states, data,_ = env.rollout(model, jr.key(eval_trial))

        rewards_trial =jnp.sum(data["reward"])
        total_rewards.append(rewards_trial)
        states = states.env_state.pipeline_state
        import jax
        states_list = [jax.tree_map(lambda x: x[i], states) for i in range(1000)]
        render = html.render(env.env.sys, states_list)
        saving_dir = project_dir + "/trial_" + str(trial) + "/visuals/"
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir, exist_ok=True)
        with open(saving_dir + "/traj_" + str(trial) + "_" + str(rewards_trial) + ".html", "w") as f:
            f.write(render)
    print(onp.mean(total_rewards))


    with open(saving_dir + "/eval.txt", "w" ) as f:
        f.write("Evaluation for 10 trials gives " + str(onp.mean(total_rewards)) + " with var " + str(onp.var(total_rewards)))



def viz_NN(project_dir, trial, config, show_dev, show_evol):
    # find saved timesteps
    step_dirs = [x for x in os.listdir(project_dir + "/trial_" + str(trial) + "/data/all_info")]
    steps = [int(el[4:]) for el in step_dirs]
    steps = [int(el) for el in steps]
    steps.sort()

    if not show_evol:
        show_gens = [steps[-1]]
    else:
        show_gens =steps

    evol_graph_features = []
    evol_hidden_features = []

    for gen in show_gens[::10]:
        show_dev_steps = list(range(config["model_config"]["dev_steps"]))

        if not show_dev:
            show_dev_steps = [show_dev_steps[-1]]

        dev_graph_features = []
        dev_hidden_features = []

        for dev_step in show_dev_steps:
            data = load_NDP_info(project_dir + "/trial_" + str(trial) + "/data/all_info/gen_" + str(gen) + "/dev_" + str(dev_step) + ".pkl")

            if show_dev:
                viz_hidden(project_dir+ "/trial_" + str(trial) + "/visuals",  gen, dev_step, data["hidden"])

                viz_network(project_dir + "/trial_" + str(trial) + "/visuals", gen, dev_step,
                            weights= data["weights"],
                                    hidden=data["hidden"],
                            mask =data["mask"],
                            n_input_nodes=config["env_config"]["obs_size"],
                            n_output_nodes=config["env_config"]["action_size"],
                            n_hidden_nodes=config["model_config"]["max_hidden_neurons"])

                viz_weights(project_dir + "/trial_" + str(trial) + "/visuals",
                            gen, dev_step, data["weights"])

            graph_features = analyze_graph(data["weights"], data["mask"])
            dev_graph_features.append(graph_features)
            hidden_features = analyze_hidden(data["hidden"])
            dev_hidden_features.append(hidden_features)

        data_save_dir = project_dir + "/trial_" + str(trial) + "/data/post_process/gen_"+ str(gen)
        if not os.path.exists(data_save_dir):
            os.makedirs(data_save_dir, exist_ok=True)

        with open(data_save_dir + "/graph_features.pkl", "wb") as f:
            pickle.dump(dev_graph_features, f)

        with open(data_save_dir + "/hidden_features.pkl", "wb") as f:
            pickle.dump(dev_hidden_features, f)

        if show_dev:
            viz_graph_growth(project_dir + "/trial_" + str(trial) + "/visuals/gen_" + str(gen), dev_graph_features)
            viz_hidden_growth(project_dir + "/trial_" + str(trial) + "/visuals/gen_" + str(gen), dev_hidden_features)

        evol_graph_features.append(dev_graph_features[-1])
        evol_hidden_features.append(dev_hidden_features[-1])

    viz_graph_growth(project_dir + "/trial_" + str(trial) + "/visuals", evol_graph_features)
    viz_hidden_growth(project_dir + "/trial_" + str(trial) + "/visuals", evol_hidden_features)



def viz_project(project_dir, show_dev, show_evol):
    #  load project config data
    with open(project_dir + "/config.yaml", "r") as f:
        config = yaml.load(f,
                        Loader=yaml.SafeLoader)

    # plot most recent best trajectory
    config["n_trials"] =1
    for trial in range(config["n_trials"]):
        #viz_trajectory(project_dir, trial, config)

        # plot neural networks
        viz_NN(project_dir, trial, config, show_dev, show_evol)


if __name__ == "__main__":
    jax.config.update('jax_platform_name', 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    random.seed(0)


    # choose the project to plot
    top_dir = "projects/for_paper/diversity/"
    #top_dir = "scripts/networks_data/NDP/"

    project_dirs = [top_dir + x for x in os.listdir(top_dir)]
    for project_dir in project_dirs:

    # only plot final trajectory and matrices
    #viz_project(project_dir, show_dev=False, show_evol=False)
        if "no_inhib" in project_dir:

            # plot developmental growth
            viz_project(project_dir, show_dev=True, show_evol=False)
