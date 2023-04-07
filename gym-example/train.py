#!/usr/bin/env python
# encoding: utf-8

from gym_example.envs.HEnv import HEnv
from ray.tune.registry import register_env
import gym
import os
import ray
import ray.rllib.agents.ppo as ppo
import shutil


def main ():
    # init directory in which to save checkpoints
    chkpt_root = "tmp/exa"
    #shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    #shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True)

    # register the custom environment
    select_env = "henv"
    register_env(select_env, lambda config: HEnv())

    # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    print(config)
    config["log_level"] = "WARN"
    agent = ppo.PPOTrainer(config, env=select_env)

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f}"
    n_iter = 100

    # train a policy with RLlib using PPO
    for n in range(n_iter):
        result = agent.train()

        print(status.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"]
                ))
        
        if (n+1) % 50 == 0:
            chkpt_file = agent.save(chkpt_root)
            print(chkpt_file)

    # examine the trained policy
    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())

if __name__ == "__main__":
    main()