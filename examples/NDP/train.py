
""" This script contains examples of how to train the NDP in different locomotion tasks. Training results saved under projects/NDP
"""

import os
import sys
sys.path.append(os.getcwd())
from src.NDP.model import make_model
from src.base.tasks.rl import BraxTask
from src.base.training.logging import Logger
from src.base.training.evolution import EvosaxTrainer, MultiESTrainer
from datetime import datetime
import yaml
import jax.random as jr
import equinox as eqx
import evosax
import jax.numpy as jnp


def run_trial(config):
    # 1.----Build the model----
    key, model_key = jr.split(config["key"])
    model = make_model(config, model_key)
    params, statics = eqx.partition(model, eqx.is_array)
    params_shaper = evosax.ParameterReshaper(params)

    # 2.----Build the task----
    env_config = config["env_config"]

    def data_fn(data: dict):
        return {}

    env = BraxTask(statics, env_config["env_name"], env_config["max_steps"],
                   env_config["backend"], data_fn=data_fn, env_kwargs={})

    # 3.----Build the trainer----
    key, train_key = jr.split(key)
    trainer_config = config["trainer_config"]

    # metrics for wandb
    def metrics_fn(state, data):
        y = {}
        y["best"] = - state.best_fitness
        y["gen_best"] = data["fitness"].max()
        y["gen_mean"] = data["fitness"].mean()
        y["gen_worse"] = data["fitness"].min()
        y["var"] = jnp.var(data["fitness"])
        return y, state.best_member, state.gen_counter, data["interm_policies"], data["best_indiv"]

    logger = Logger(bool(trainer_config["log"]), metrics_fn=metrics_fn, ckpt_dir=config["trial_dir"] + "/data",
                    ckpt_freq=config["ckpt_freq"], dev_steps=config["model_config"]["dev_steps"])

    fitness_shaper = evosax.FitnessShaper(maximize=True, centered_rank=bool(trainer_config.get("centered_rank", False)))
    if isinstance(trainer_config["strategy"], list):
        trainers = []
        for i, strat in enumerate(trainer_config["strategy"]):
            if strat == "LGA":
                net_ckpt_path = trainer_config.get("net_ckpt_path", "datasets/2023_04_lga.pkl")
                strat = evosax.LGA(popsize=trainer_config["popsize"] if isinstance(trainer_config["popsize"], int) else
                trainer_config["popsize"][i],
                                   num_dims=params_shaper.total_params,
                                   net_ckpt_path=net_ckpt_path)
            elif strat == "LES":
                net_ckpt_path = trainer_config.get("net_ckpt_path", "datasets/2023_10_les_v2.pkl")
                strat = evosax.LES(popsize=trainer_config["popsize"] if isinstance(trainer_config["popsize"], int) else
                trainer_config["popsize"][i],
                                   num_dims=params_shaper.total_params,
                                   net_ckpt_path=net_ckpt_path)
            trainer = EvosaxTrainer(
                train_steps=trainer_config["generations"] if isinstance(trainer_config["generations"], int) else
                trainer_config["generations"][i],
                task=env,
                strategy=strat,
                params_shaper=params_shaper,
                popsize=trainer_config["popsize"] if isinstance(trainer_config["popsize"], int) else
                trainer_config["popsize"][i],
                fitness_shaper=fitness_shaper,
                logger=logger,
                progress_bar=True)
            trainers.append(trainer)
        trainer = MultiESTrainer(trainers)
    else:
        if trainer_config["strategy"] == "LGA":
            net_ckpt_path = trainer_config.get("net_ckpt_path", "datasets/2023_04_lga.pkl")
            strategy = evosax.LGA(popsize=trainer_config["popsize"],
                                  num_dims=params_shaper.total_params,
                                  net_ckpt_path=net_ckpt_path)
        elif trainer_config["strategy"] == "LES":
            net_ckpt_path = trainer_config.get("net_ckpt_path", "datasets/2023_10_les_v2.pkl")
            strategy = evosax.LES(popsize=trainer_config["popsize"],
                                  num_dims=params_shaper.total_params,
                                  net_ckpt_path=net_ckpt_path)
        else:
            strategy = trainer_config["strategy"]
        trainer = EvosaxTrainer(train_steps=trainer_config["generations"],
                                task=env,
                                strategy=strategy,
                                params_shaper=params_shaper,
                                popsize=trainer_config["popsize"],
                                fitness_shaper=fitness_shaper,
                                logger=logger,
                                progress_bar=True,
                                n_devices=trainer_config["n_devices"],
                                eval_reps=trainer_config["eval_reps"])

    logger.wandb_init(project=config["wandb_project"],
                      name=config["trial_dir"],
                      config=config)
    final_state = trainer.init_and_train_(train_key)
    logger.wandb_finish()
    return final_state


def train(env_name, inhibition, intrinsic, num_hidden_neurons, generations, strategy="DES", seed=24109):
    project_dir = "projects/gecco_2024/" + datetime.today().strftime(
        '%Y_%m_%d') + "/" + env_name + "/inhibition_" + inhibition + "_intrinsic_" + str(intrinsic) + "_nneurons_" + str(num_hidden_neurons)

    if not os.path.exists(project_dir):
        os.makedirs(project_dir, exist_ok=True)


    # ----- prepare project configuration ------
    with open("examples/NDP/default_config.yaml", "r") as f:
        default_config = yaml.load(f, Loader=yaml.SafeLoader)

    with open("examples/envs_info.yaml", "r") as f:
        envs_info = yaml.load(f, Loader=yaml.SafeLoader)

    config = default_config
    config["model_config"] = {"inhibition": inhibition,
                              "intrinsic": intrinsic,
                              "algorithm": "NDP",
                              "max_hidden_neurons": num_hidden_neurons,
                              "dev_steps": 15,
                              "inhibit_for": 2}

    config["env_config"]["action_size"] = envs_info[env_name]["output"]
    config["env_config"]["obs_size"] = envs_info[env_name]["input"]
    config["env_config"]["env_name"] = env_name
    config["trainer_config"]["strategy"] = strategy
    config["trainer_config"]["generations"] = generations

    config["seed"] = seed
    config["project_dir"] = project_dir
    with open(config["project_dir"] + "/config.yaml", "w") as f:
        yaml.dump(config, f)

    # -----------------------------------------
    key = jr.PRNGKey(config["seed"])
    for trial in range(n_trials):
        key, _ = jr.split(key)
        config["key"] = key
        config["trial_dir"] = project_dir + "/trial_" + str(trial)

        if not os.path.exists(config["trial_dir"]):
            os.makedirs(config["trial_dir"], exist_ok=True)

        run_trial(config)

def NDP():
    train(env_name="ant", inhibition="none", intrinsic=True, num_hidden_neurons=64, strategy="DES", generations=5000)
    train(env_name="reacher", inhibition="none", intrinsic=True, num_hidden_neurons=64, strategy="DES", generations=7000)
    train(env_name="inverted_double_pendulum", inhibition="none", intrinsic=False, num_hidden_neurons=64, strategy="DES", generations=1000)
    train(env_name="halfcheetah", inhibition="none", intrinsic=True, num_hidden_neurons=64, strategy="DES", generations=10000)

def NDP_inhib():
    train(env_name="ant", inhibition="hidden_and_mitosis", intrinsic=False, num_hidden_neurons=64, strategy="DES", generations=1000)
    train(env_name="reacher", inhibition="hidden_and_mitosis", intrinsic=False, num_hidden_neurons=64, strategy="DES",
          generations=1000)
    train(env_name="halfcheetah", inhibition="hidden_and_mitosis", intrinsic=False, num_hidden_neurons=64, strategy="DES",
          generations=10000)
    train(env_name="halfcheetah", inhibition="hidden_and_mitosis", intrinsic=False, num_hidden_neurons=64, strategy="DES",
          generations=10000)

def NDP_vanilla():
    train(env_name="ant", inhibition="none", intrinsic=False, num_hidden_neurons=64, strategy="DES", generations=5000)
    train(env_name="reacher", inhibition="none", intrinsic=False, num_hidden_neurons=64, strategy="DES", generations=7000)
    train(env_name="inverted_double_pendulum", inhibition="none", intrinsic=False, num_hidden_neurons=64, strategy="DES", generations=1000)
    train(env_name="halfcheetah", inhibition="none", intrinsic=False, num_hidden_neurons=64, strategy="DES", generations=10000)



if __name__ == "__main__":
    n_trials = 3
    NDP()
    NDP_inhib()
    NDP_vanilla()



    #NDP_noinhib()
