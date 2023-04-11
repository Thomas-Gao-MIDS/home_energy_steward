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

from utils import maybe_rescale_box_space, to_raw, to_scaled

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class HEnv(gym.Env):

    def __init__(self,
                 scen_id: str = "001",
                 # rescale_spaces doesn't seem to be working correctly. ev_action got messed up.
                 rescale_spaces: bool = False):
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

        self.rescale_spaces = rescale_spaces
        self.es_action_last = 0
        self.ev_action_last = 0
        self.cum_engy_unused = 0

        self.obs_labels = ['grid_cost', 'pv_power', 'dev_power',
                           'es_storage', 'ev_in', 'ev_energy_required', 
                           'pv_excess', 'es_action_last', 'ev_action_last']
        obs_low = np.zeros((len(self.obs_labels),), dtype=np.float32)
        obs_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0])
        obs_high = np.array([1.0, 10.0, 10.0,
                             50.0, 1.0, 50.0, 10.0, 1.0, 1.0], dtype=np.float32)
        self._observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.observation_space = maybe_rescale_box_space(
            self._observation_space, rescale = self.rescale_spaces)
        
        # action space
        self.act_labels = ['es_action', 'ev_action']
        act_low = np.array([-1.0, 0.0], dtype=np.float32)
        act_high = np.array([1.0, 1.0], dtype=np.float32)
        self._action_space = gym.spaces.Box(low=act_low, high=act_high, dtype=np.float32)
        self.action_space = maybe_rescale_box_space(
            self._action_space, rescale=self.rescale_spaces)

        # initialize
        self.seed(seed=1338)
        self.reset()
    
    def reset(self, rand=False):
        """
        Reset environment to initial state
        Return observation of initial state
        """

        self.simulation_step = 0
        self.es_storage = self.env_config['components'][1]['config']['init_storage']
        self.ev_energy_required = self.ev_profile_data['energy_required_kwh'][0]
        if(rand):
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
        self.reward = 0
        self.es_action_last = 0
        self.ev_action_last = 0
        
        return self.current_obs
    
    def get_obs(self):

        raw_obs = np.array([self.grid_costs[self.simulation_step], 
                        self.pv_powers[self.simulation_step], 
                        self.dev_powers[self.simulation_step],
                        self.es_storage, self.get_ev_in(), self.ev_energy_required,
                        max(self.pv_powers[self.simulation_step]-self.dev_powers[self.simulation_step],0), 
                        self.es_action_last, 
                        self.ev_action_last],
                        dtype=np.float32)
        
        if self.rescale_spaces:
            obs = to_scaled(raw_obs, self._observation_space.low, self._observation_space.high)
        else:
            obs = raw_obs
        
        return obs

    def step(self, action):
        """
        Performs transition step
        Return next observation, reward, done, additional info
        """
        # Actions
        if self.rescale_spaces:
            action = to_raw(action, self._action_space.low, self._action_space.high)

        es_action = action[0] # <0 ~ supply thru discharge; >0 ~ charge 
        es_action = max(min(es_action, 1), -1)
        ev_action = action[1]
        ev_action = max(min(ev_action, 1), 0)

        pw_to_engy = self.minutes_per_step / 60

        # Energy Consumption (+)
        dev_engy = self.dev_powers[self.simulation_step] * pw_to_engy
        
        ev_in = self.get_ev_in()
        ev_engy = ev_in * ev_action * self.ev_max_power * pw_to_engy
        ev_engy = min(ev_engy, self.ev_energy_required)
        self.ev_energy_required -= ev_engy

        es_engy_c = 0
        if es_action >= 0: # charge
            es_engy_c = es_action * self.es_max_power * pw_to_engy
            es_engy_c = min(es_engy_c, (self.es_range[1] - self.es_storage) / self.es_charge_efficiency)
            self.es_storage += es_engy_c * self.es_charge_efficiency

        engy_consumption = dev_engy + ev_engy + es_engy_c

        # Energy Supply (-)
        pv_engy = - self.pv_powers[self.simulation_step] * pw_to_engy
        
        es_engy_s = 0
        if es_action < 0: # supply thru discharge
            es_engy_s = es_action * self.es_max_power * pw_to_engy
            es_engy_s = -min((self.es_storage-self.es_range[0]) * self.es_discharge_efficiency, -es_engy_s)
            es_engy_s = -min(-es_engy_s, engy_consumption+pv_engy) # supply depends on contemperaneous energy. clip extra supply
            self.es_storage += es_engy_s / self.es_discharge_efficiency

        engy_supply = pv_engy + es_engy_s

        # Net Energy
        net_engy = engy_consumption + engy_supply
        if net_engy >= 0: 
            engy_unused = 0
            grid_engy = -net_engy
        else: 
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
        self.es_engy = es_engy_c if es_engy_c > 0 else es_engy_s
        self.ev_engy = ev_engy
        self.grid_engy = grid_engy
        self.engy_consumption = engy_consumption
        self.engy_supply = engy_supply
        self.engy_unused = engy_unused
        self.ecost = ecost
        self.es_action_last = self.es_engy / (self.es_max_power * pw_to_engy)
        self.ev_action_last = self.ev_engy / (self.ev_max_power * pw_to_engy)

        # Compute done & next obs
        done = False
        if self.simulation_step+1 == self.max_episode_steps - 24:
            self.reward = - self.ev_energy_required**2
        if self.simulation_step+1 == self.max_episode_steps:
            done = True
            self.reward = - self.cum_ecost - self.ev_energy_required**2
            
            print("ecost:", round(self.cum_ecost,1), 
                  "| ev_required (0):", round(self.ev_energy_required,1), 
                  "| engy_unused (0):", round(self.cum_engy_unused,1),
                  "| es_storage (1)", round(self.es_storage,1))
            
            
        else:
            self.simulation_step += 1
            next_obs = self.get_obs()
            self.current_obs = next_obs

        return self.current_obs, self.reward, done, {}

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
        obs = henv.reset(rand=False)
        action = [-1, 1]
        print(action)
        while True:
            #action = henv.action_space.sample()
            obs, r, done, _ = henv.step(action)
            if done:
                break