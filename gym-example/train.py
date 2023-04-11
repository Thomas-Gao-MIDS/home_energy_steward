#!/usr/bin/env python
# encoding: utf-8

from gym_example.envs.HEnv import HEnv
from ray.tune.registry import register_env
import gym
import os
import ray
import ray.rllib.agents.ppo as ppo
import shutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main ():

    scen_id = '001'
    n_iter = 300

    # init directory in which to save checkpoints
    chkpt_root = "tmp/exa/"+scen_id
    #shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    #shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True, local_mode=False)

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
    #config['seed'] = 123

    agent = ppo.PPOTrainer(config, env=select_env)

    status = "{:2d} reward {:6.2f}"
    rewards = []
    max_episode_reward = -1e10
    chkpt_file = None
    # train a policy with RLlib using PPO
    for n in range(n_iter):
        result = agent.train()

        print(status.format(n + 1, result["episode_reward_mean"]))
        rewards.append(result["episode_reward_mean"])
        
        #save checkpoint if reward is better than previous
        if result["episode_reward_mean"] >= max_episode_reward:
            max_episode_reward = result["episode_reward_mean"]
            #delete previous checkpoint
            if chkpt_file:
                print('removing previous checkpoint file: ', chkpt_file)
                #remove the folder up to the last '/'
                os.system('rm -rf '+'/'.join(chkpt_file.split('/')[:-1]))
            #save new checkpoint
            chkpt_file = agent.save(chkpt_root)
            print(chkpt_file)
        #if n_iter - (n+1) <10:
        #    chkpt_file = agent.save(chkpt_root)
        #    print(chkpt_file)

    # examine the trained policy
    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())

    with PdfPages('output/'+scen_id+'_rewards.pdf') as pdf:
        plt.figure()

        plt.plot(rewards)
        plt.title('Rewards');
        pdf.savefig(); plt.close()

        plt.plot(rewards[50:])
        plt.title('Rewards After 50 Iters');
        pdf.savefig(); plt.close()
    

if __name__ == "__main__":
    main()