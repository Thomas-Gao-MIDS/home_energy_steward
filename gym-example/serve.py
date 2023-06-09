#!/usr/bin/env python
# encoding: utf-8

from gym_example.envs.HEnv import HEnv
from ray.tune.registry import register_env
import gym
import os
import ray
import ray.rllib.agents.ppo as ppo
import pandas as pd

def main ():
 
    scen_id = '001'
    model_iter = '300'

    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True, local_mode=True)

    # register the custom environment
    select_env = "henv"
    register_env(select_env, lambda config: HEnv())

    # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["rollout_fragment_length"] = 288
    config["train_batch_size"] = 288 * 16
    config['lr_schedule'] = [[0, 2e-3],[250*288,1e-4]]
    config['batch_mode'] = "complete_episodes"
    config['seed'] = 123


    agent = ppo.PPOTrainer(config, env=select_env)

    chkpt_file = 'tmp/exa/'+scen_id+'/checkpoint_000'+model_iter+'/checkpoint-'+model_iter
    print(chkpt_file)

    # apply the trained policy in a rollout
    agent.restore(chkpt_file)
    env = gym.make(select_env, scen_id=scen_id)

    state = env.reset(rand=False)
    sum_reward = 0
    n_step = 288

    # extract data
    timestamps = []
    es_action = []
    ev_action = []
    pv_engy = []
    dev_engy = []
    es_engy = []
    ev_engy = []
    grid_engy = []
    engy_consumption = []
    engy_supply = []
    engy_unused = []
    cum_engy_unused = []
    grid_cost = []
    ecost = []
    cum_ecost = []
    ev_energy_required = []
    es_storage = []

    for step in range(n_step):
        action = agent.compute_action(state)
        #print(action)
        state, reward, done, info = env.step(action)
        sum_reward += reward
        
        timestamps.append(env.timestamps[step])
        #es_action.append(max(min(action[0], 1),-1))
        #ev_action.append(max(min(action[1], 1), 0))
        es_action.append(env.es_action_last)
        ev_action.append(env.ev_action_last)
        pv_engy.append(env.pv_engy)
        dev_engy.append(env.dev_engy)
        es_engy.append(env.es_engy)
        ev_engy.append(env.ev_engy)
        grid_engy.append(env.grid_engy)
        engy_consumption.append(env.engy_consumption)
        engy_supply.append(env.engy_supply)
        engy_unused.append(env.engy_unused)
        cum_engy_unused.append(env.cum_engy_unused)
        grid_cost.append(env.grid_costs[step])
        ecost.append(env.ecost)
        cum_ecost.append(env.cum_ecost)
        ev_energy_required.append(env.ev_energy_required)
        es_storage.append(env.es_storage)

        if done == 1:
            # report at the end of each episode
            print("cumulative reward", sum_reward)
            state = env.reset(rand=False)
            sum_reward = 0

            df = \
            pd.DataFrame(list(zip(timestamps, 
                                  es_action, ev_action,
                                  pv_engy, dev_engy, es_engy, ev_engy, grid_engy,
                                  engy_consumption, engy_supply, engy_unused, cum_engy_unused,
                                  grid_cost, ecost, cum_ecost,
                                  ev_energy_required, es_storage)),
                                  columns = ["timestamps", 
                                  "es_action", "ev_action",
                                  "pv_engy", "dev_engy", "es_engy", "ev_engy", "grid_engy",
                                  "engy_consumption", "engy_supply", "engy_unused", "cum_engy_unused",
                                  "grid_cost", "ecost", "cum_ecost",
                                  "ev_energy_required", "es_storage"])
            df.to_csv("output/"+scen_id+"_data.csv", index=False)

            

if __name__ == "__main__":
    main()