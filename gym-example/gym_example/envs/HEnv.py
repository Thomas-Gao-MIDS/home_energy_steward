import json
import os
import pandas as pd
import numpy as np

import gym
from gym.utils import seeding
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents import ppo

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# states, actions, state transition, reward function

class HEnv(gym.Env):

    def __init__(self,
                 scen_id: str = "002"):
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

        # check battery - why always dischrarge immediately after charging? check code!
        # try rescale observation
        # implement reset to change scenario, entropy_coeff
        # for generalization, introduce weather, temperature, day
        # pv_excess is max(pv_power - dev_power, 0), for signaling es_action to charge
        self.obs_labels = ['grid_cost', 'pv_power', 'dev_power',
                           'es_charge', 'ev_in', 'ev_energy_required', 'pv_excess']
        obs_low = np.zeros((len(self.obs_labels),), dtype=np.float32)
        obs_high = np.array([1.0, 10.0, 10.0,
                             50.0, 1.0, 50.0, 10.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # initial
        self.seed(seed=1338)
        self.reset()
    
    def reset(self, train=True):
        """
        Reset environment to initial state
        Return observation of initial state
        """

        self.simulation_step = 0
        self.es_storage = self.env_config['components'][1]['config']['init_storage']
        self.ev_energy_required = self.ev_profile_data['energy_required_kwh'][0]
        if(train):
            self.es_storage += self.np_random.rand()
            self.ev_energy_required -= self.np_random.rand()

        self.current_obs = self.get_obs()
        self.cum_ecost = 0
        self.cum_engy_unused = 0

        self.pv_engy = 0
        self.dev_engy = 0
        self.es_engy = 0
        self.ev_engy = 0
        self.grid_engy = 0
        self.engy_consumption = 0
        self.engy_supply = 0
        self.engy_unused = 0
        self.ecost = 0
        
        return self.current_obs
    
    def get_obs(self):

        if(self.simulation_step == self.max_episode_steps):
            obs = np.array([0,0,0,self.es_storage,0,self.ev_energy_required,0], 
                           dtype=np.float32)
        else:
            obs = np.array([self.grid_costs[self.simulation_step], 
                            self.pv_powers[self.simulation_step], 
                            self.dev_powers[self.simulation_step],
                            self.es_storage, self.get_ev_in(), self.ev_energy_required,
                            max(self.pv_powers[self.simulation_step]-self.dev_powers[self.simulation_step],0)], 
                            dtype=np.float32)
        
        return obs

    def step(self, action):
        """
        Performs transition step
        Return next observation, reward, done, additional info
        Issue: 
            at training, es does not learn to charge from unused solar
                seems difficult to learn. how to fix?
                * try different algos
            at inference, es & ev do not charge at all
                * get in-sample results out using callback
                why & how to fix?
                * train multiple scenarios
        Next Steps:
            Mult scenarios
            Baseline check es_initial charge & es_efficiency
            
        """
        # Actions
        es_action = action[0] # <0 ~ supply thru discharge; >0 ~ charge 
        es_action = max(min(es_action, 1), -1)
        ev_action = action[1]
        ev_action = max(min(ev_action, 1), 0)

        pw_to_engy = self.minutes_per_step / 60

        # Energy Consumption: +
        dev_engy = self.dev_powers[self.simulation_step] * pw_to_engy
        
        ev_in = self.get_ev_in()
        ev_engy = ev_in * ev_action * self.ev_max_power * pw_to_engy
        ev_engy = min(ev_engy, self.ev_energy_required)
        self.ev_energy_required -= ev_engy

        es_engy = 0
        if es_action >= 0: # charge
            es_engy = es_action * self.es_max_power * pw_to_engy
            es_engy = min(es_engy, (self.es_range[1] - self.es_storage) / self.es_charge_efficiency)
            self.es_storage += es_engy * self.es_charge_efficiency

        engy_consumption = dev_engy + ev_engy + es_engy

        # Energy Supply: -
        pv_engy = - self.pv_powers[self.simulation_step] * pw_to_engy
        
        if es_action < 0: # supply thru discharge
            es_engy = es_action * self.es_max_power * pw_to_engy
            es_engy = -min((self.es_storage-self.es_range[0]) * self.es_discharge_efficiency, -es_engy)
            self.es_storage += es_engy / self.es_discharge_efficiency

        engy_supply = pv_engy + es_engy

        net_engy = engy_consumption + engy_supply
        if net_engy >= 0: 
            engy_unused = 0
            grid_engy = -net_engy
        else: 
            grid_engy = 0 
            engy_unused = -net_engy
            self.cum_engy_unused += engy_unused
            grid_engy = 0

        engy_supply += grid_engy

        # Electricity Cost
        ecost = - grid_engy * self.grid_costs[self.simulation_step]
        self.cum_ecost += ecost

        # Save Variables
        self.pv_engy = pv_engy
        self.dev_engy = dev_engy
        self.es_engy = es_engy
        self.ev_engy = ev_engy
        self.grid_engy = grid_engy
        self.engy_consumption = engy_consumption
        self.engy_supply = engy_supply
        self.engy_unused = engy_unused
        self.ecost = ecost

        # Compute reward
        reward = (-ecost)
        if self.simulation_step+1 == self.max_episode_steps:
            reward -= (1.0 * self.ev_energy_required**2)
            #reward -= (0.1 * self.cum_engy_unused**2)
            #reward -= (0.05 * self.es_storage**2)

        # Compute done & next obs
        done = False
        if self.simulation_step+1 == self.max_episode_steps:
            done = True
            print("ecost:", round(self.cum_ecost), 
                  "| engy_unused (0):", round(self.cum_engy_unused),
                  "| ev_required (0):", round(self.ev_energy_required), 
                  "| es_storage (1)", round(self.es_storage))
        else:
            self.simulation_step += 1
            next_obs = self.get_obs()
            self.current_obs = next_obs

        return self.current_obs, reward, done, {}

    def get_ev_in(self):
        
        ev_in = 0

        if self.simulation_step * self.minutes_per_step >= self.ev_start_time and \
            self.simulation_step * self.minutes_per_step <= self.ev_end_time:
            ev_in=1

        return ev_in
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


if __name__ == "__main__":
    henv = HEnv()
    for _ in range(1):
        obs = henv.reset()
        while True:
            action = henv.action_space.sample()
            obs, r, done, _ = henv.step(action)
            if done:
                break