import json
import os
import pandas as pd
import numpy as np

import gym
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents import ppo

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# states, actions, state transition, reward function

class HEnv(gym.Env):

    def __init__(self,
                 scen_id: str = "001"):
        """
        self.observation_space, self.action_space
        """
        # Define action space: bounds, space type, shape

        with open(os.path.join(THIS_DIR, "data/"+scen_id+".json"), 'r') as f:
            env_config = json.load(f)

        self.env_config = env_config    
        self.max_episode_steps = env_config['max_episode_steps']
        self.minutes_per_step = 5
        self.timestamps = env_config['timestamps']
        self.grid_costs = env_config['grid_cost']
        self.pv_powers = env_config['components'][0]['config']['profile_data']
        self.es_max_power = env_config['components'][1]['config']['max_power']
        self.es_range = env_config['components'][1]['config']['storage_range']
        # <- converged to local minimal of not using battery
        # introduce random starting level? keep inference steady
        # or do more exploration? or more training?
        # penalize excess energy in the end?
        # also, ev not charging. will need penalty in the end
        self.es_storage = env_config['components'][1]['config']['init_storage']
        self.es_charge_efficiency = env_config['components'][1]['config']['charge_efficiency']
        self.es_discharge_efficiency = env_config['components'][1]['config']['discharge_efficiency']
        self.ev_max_power = env_config['components'][2]['config']['max_charge_rate_kw']
        self.ev_profile_data = pd.read_json(json.dumps(env_config['components'][2]['config']['profile_data']), orient='split')
        self.ev_energy_required = self.ev_profile_data['energy_required_kwh'][0]
        self.ev_start_time = self.ev_profile_data['start_time_min'][0]
        self.ev_end_time = self.ev_profile_data['end_time_park_min'][0]
        self.dev_profile_data = env_config['components'][3]['config']['profile_data']
        self.dev_powers = np.array(self.dev_profile_data['hvac_power'])+np.array(self.dev_profile_data['other_power']).tolist()

        # action space
        self.act_labels = ['es_action', 'ev_action']
        act_low = np.array([-1.0, 0.0], dtype=np.float32)
        act_high = np.array([1.0, 1.0], dtype=np.float32)
        self.action_space = gym.spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        # observation space
        # for generalization, introduce weather, temperature, day
        self.obs_labels = ['grid_cost', 'pv_power', 'dev_power',
                           'es_charge', 'ev_in', 'ev_energy_required']
        obs_low = np.zeros((len(self.obs_labels),), dtype=np.float32)
        obs_high = np.array([max(self.grid_costs), max(self.pv_powers), max(self.dev_powers),
                             self.es_range[1], 1.0, self.ev_energy_required], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        

        # initial
        self.simulation_step = 0
        self.current_obs = self.get_obs()
        self.cum_ecost = 0
        self.cum_excess_engy = 0
        
    
    def reset(self):
        """
        Reset environment to initial state
        Return observation of initial state
        """

        self.simulation_step = 0
        self.es_storage = self.env_config['components'][1]['config']['init_storage']
        self.ev_energy_required = self.ev_profile_data['energy_required_kwh'][0]

        self.current_obs = self.get_obs()
        self.cum_ecost = 0
        self.cum_excess_engy = 0

        self.pv_engy = 0
        self.dev_engy = 0
        self.es_engy = 0
        self.ev_engy = 0
        self.grid_engy = 0
        self.engy_consumption = 0
        self.engy_supply = 0
        self.energy_excess = 0
        self.ecost = 0
        
        return self.current_obs
    
    def get_obs(self):

        if(self.simulation_step == self.max_episode_steps):
            obs = np.array([0,0,0,self.es_storage,0,self.ev_energy_required], 
                           dtype=np.float32)
        else:
            obs = np.array([self.grid_costs[self.simulation_step], 
                            self.pv_powers[self.simulation_step], 
                            self.dev_powers[self.simulation_step],
                            self.es_storage, self.get_ev_in(), self.ev_energy_required], 
                            dtype=np.float32)
        
        return obs

    def step(self, action):
        """
        Performs transition step
        Return next observation, reward, done, additional info
        """
        # Actions
        es_action = action[0] # <0 ~ supply thru discharge; >0 ~ charge 
        es_action = max(min(es_action, 1), -1)
        ev_action = action[1]
        ev_action = max(min(ev_action, 1), 0)
        
        # Energy Consumption
        pw_to_engy = self.minutes_per_step / 60

        dev_engy = self.dev_powers[self.simulation_step] * pw_to_engy
        
        ev_in = self.get_ev_in()
        ev_engy = ev_in * ev_action * self.ev_max_power * pw_to_engy
        ev_engy = min(ev_engy, self.ev_energy_required)
        self.ev_energy_required -= ev_engy

        engy_consumption = dev_engy + ev_engy

        if es_action >= 0: # charge
            es_engy = es_action * self.es_max_power * pw_to_engy
            es_engy = min(es_engy, (self.es_range[1] - self.es_storage) / self.es_charge_efficiency)
            self.es_storage += es_engy * self.es_charge_efficiency
            engy_consumption += es_engy

        # Energy Supply
        pv_engy = self.pv_powers[self.simulation_step] * pw_to_engy
        engy_supply = pv_engy
        
        if es_action < 0: # supply thru discharge
            es_engy = es_action * self.es_max_power * pw_to_engy
            es_engy = max(-(self.es_storage-self.es_range[0]) * self.es_discharge_efficiency, es_engy)
            self.es_storage += es_engy / self.es_charge_efficiency
            engy_supply -= es_engy

        print(es_action, es_engy, self.es_storage)

        if engy_consumption > engy_supply:
            grid_engy = engy_consumption - engy_supply
            engy_excess = 0
        else:
            grid_engy = 0
            engy_excess = engy_supply - engy_consumption
            self.cum_excess_engy += engy_excess

        # Electricity Cost
        ecost = grid_engy * self.grid_costs[self.simulation_step]
        self.cum_ecost += ecost

        # Save Variables
        self.pv_engy = -pv_engy
        self.dev_engy = dev_engy
        self.es_engy = es_engy
        self.ev_engy = ev_engy
        self.grid_engy = -grid_engy
        self.engy_consumption = engy_consumption
        self.engy_supply = engy_supply
        self.engy_excess = -engy_excess
        self.ecost = ecost

        # Compute reward
        reward = - (ecost + 0.2 * engy_excess**2) 
        if self.simulation_step == self.max_episode_steps:
            reward -= 0.2 * self.ev_energy_required**2

        # Compute done & next obs
        self.simulation_step += 1
        done = False
        if self.simulation_step == self.max_episode_steps:
            done = True
            print(round(self.cum_ecost, 3), round(self.cum_excess_engy, 3))

        next_obs = self.get_obs()

        self.current_obs = next_obs

        return self.current_obs, reward, done, {}

    def get_ev_in(self):
        
        ev_in = 0

        if self.simulation_step * self.minutes_per_step >= self.ev_start_time and \
            self.simulation_step * self.minutes_per_step <= self.ev_end_time:
            ev_in=1

        return ev_in


if __name__ == "__main__":
    henv = HEnv()
    for _ in range(1):
        obs = henv.reset()
        while True:
            action = [0.05, 0.0]
            obs, r, done, _ = henv.step(action)
            if done:
                break


"""
def env_creator(env_config):
    return HENV()

class Trainer():
    def __init__(self):
        ray.init()
        register_env("henv", env_creator)  
        self.config = {"env": 'henv',
                       "env_config": {},
                       } 
        self.env_class = 'henv'
        self.agent = None
    
    def train(self):
        analysis = tune.run("PPO",
                            local_dir = os.path.join(THIS_DIR, "output"),
                            checkpoint_at_end=True,
                            stop={'training_iteration': 2},
                            config=self.config,
                            verbose = 0)

        trial_logdir = analysis.get_best_logdir(metric="episode_reward_mean", mode="max")  
        path = analysis.get_best_checkpoint(trial_logdir, metric="episode_reward_mean", mode="max")
        print(path)

        return path, analysis    
    
    def load(self, path):
        self.agent = ppo.PPOTrainer(config=self.config, env=self.env_class)
        self.agent.restore(path)

    def test(self):
        env = HENV()

        # run until episode ends
        episode_reward = 0
        done = False
        obs = env.reset()
        while not done:
            action = self.agent.compute_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

            print(action,
                env.es_storage,
                env.ev_energy_required,
                env.cum_ecost,
                env.cum_excess_engy)

        return episode_reward
"""

"""
from ray.rllib.utils.test_utils import check_compute_single_action
trainer = Trainer()
path, _ = trainer.train()
path = '/Users/chengchungao/w210/PowerGridworld/simplify/output/PPO/PPO_henv_03c02_00000_0_2023-04-06_00-59-54/checkpoint_000100/checkpoint-100'
trainer.load(path)
trainer.test()
"""


"""
if False:
    ray.init()
    register_env("henv", env_creator) 
    tune.run("PPO",
            local_dir = os.path.join(THIS_DIR, "output"),
            checkpoint_at_end=True,
            stop={'training_iteration': 10},
            config={"env": 'henv',
                    "env_config":{},
                    })

if True:
    from ray.rllib.agents import ppo
    agent = ppo.PPOTrainer(config={"env": "henv", "env_config":{}})

"""

#print(henv.cum_ecost, henv.cum_excess_engy,
#      henv.es_storage, henv.ev_energy_required)

