from simglucose.simulation.env import T1DSimEnv_MARL as _T1DSimEnv_MARL
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action
import numpy as np
import pandas as pd
import os
import glob
import pkg_resources
import gym
from gym import spaces
from gym.utils import seeding
from datetime import datetime
import gymnasium
from copy import copy
from gymnasium.spaces import Discrete
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo import AECEnv
import shutil
from typing import Any, Dict, TypeVar

ObsType = TypeVar("ObsType")

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    "simglucose", "params/vpatient_params.csv"
)

n_possible_actions = 11  # Necessary for maskable PPO


class T1DSimEnv_MARL(gym.Env):
    metadata = {"render_modes": ["human"],
                'name': 'T1DSimGymnasiumEnv_MARL'
                }
    SENSOR_HARDWARE = "Dexcom"
    INSULIN_PUMP_HARDWARE = "Insulet"

    def __init__(self, patient_name=None, custom_scenario=None, reward_fun=None,
                 seed=None, training=None, folder=None):
        if patient_name is None:
            patient_name = ["adolescent#001"]

        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.training = training
        self.folder = folder
        self.np_random, _ = seeding.np_random(seed=seed)
        self.custom_scenario = custom_scenario
        self.env, _, _, _ = self._create_env()

    def _step(self, action: float):
        act = Action(basal=action, bolus=0)
        if self.reward_fun is None:
            obs, reward, done, info = self.env.step(act)
        else:
            obs, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
        return obs, reward, done, False, info

    def _raw_reset(self):
        return self.env.reset()

    def _reset(self):
        self.env, _, _, _ = self._create_env()
        obs, _, _, _ = self.env.reset()
        return obs

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env()
        return [seed1, seed2, seed3, seed4]

    def _create_env(self):
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31
        seed4 = seeding.hash_seed(seed3 + 1) % 2**31

        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)

        if isinstance(self.patient_name, list):
            patient_name = self.np_random.choice(self.patient_name)
            patient = T1DPatient.withName(patient_name, random_init_bg=True, seed=seed4)
        else:
            patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)

        if isinstance(self.custom_scenario, list):
            scenario = self.np_random.choice(self.custom_scenario)
        else:
            scenario = (RandomScenario(start_time=start_time, seed=seed3)
                        if self.custom_scenario is None
                        else self.custom_scenario)

        reward_fun = self.reward_fun
        training = self.training
        folder = self.folder

        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv_MARL(patient, sensor, pump, scenario, reward_fun, training, folder)
        return env, seed2, seed3, seed4

    def _render(self, mode="human", close=False):
        self.env.render(close=close)

    def show_history(self):
        self.env.show_history()

    def _close(self):
        super()._close()
        self.env._close_viewer()

    @property
    def action_space(self):
        ub = self.env.pump._params["max_basal"]
        return spaces.Box(low=0, high=ub, shape=(1,))
        # return spaces.discrete.Discrete(n_possible_actions)

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1000, shape=(1,))

    @property
    def max_basal(self):
        return self.env.pump._params["max_basal"]


class T1DSimGymnasiumEnv_MARL(AECEnv):
    metadata = {"render_modes": ["human"],
                'name': 'T1DSimGymnasiumEnv_MARL'
                }

    MAX_BG = 1000

    def __init__(self, patient_name=None, custom_scenario=None, reward_fun=None,
                 seed=None, render_mode=None, training=None, folder=None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.env = T1DSimEnv_MARL(
            patient_name=patient_name,
            custom_scenario=custom_scenario,
            reward_fun=reward_fun,
            seed=seed,
            training=training,
            folder=folder
        )
        
        self.agents = ["Jerry", "Morty", "Rick"]
        self.possible_agents = self.agents[:]
        
        # self.agent_selection = None
        self.step_num = 1
        self.iper_s = 120
        self.ipo_s = 85

        if self.env.training:
            current_time = datetime.now()
            self.time_suffix_Min = current_time.strftime("%Y%m%d_%H%M")
            self.folder = folder
            self.subfolder_train = f"training_{self.env.patient_name}_{self.time_suffix_Min}"
            os.makedirs(os.path.join(self.folder, self.subfolder_train), exist_ok=True)
            
            data_diz = {
                'step': [],
                'CGM': [],
                'Action': [],
                'Active agent': [],
                'Jerry_Reward': [],
                'Morty_Reward': [],
                'Rick_Reward': [],
                'Total_Reward': [],
                'Jerry_Termination': [],
                'Morty_Termination': [],
                'Rick_Termination': [],
                'Jerry_Truncation': [],
                'Morty_Truncation': [],
                'Rick_Truncation': [],
                'Infos': []
            }
        
            self.df_training = pd.DataFrame(data_diz)
        
        
        # self.action_space = gymnasium.spaces.discrete.Discrete(n_possible_actions)
        self.action_spaces = {i: gymnasium.spaces.discrete.Discrete(n_possible_actions) for i in self.agents}
        self.observation_spaces = {i:gymnasium.spaces.Dict(
            {
                "observation": gymnasium.spaces.Box(
                    low=0, high=self.MAX_BG, shape=(1,), dtype=np.float32
                ),
                # "action_mask": gymnasium.spaces.Box(
                    # low=0, high=self.env.max_basal, shape=(1,), dtype=np.float32
                # ),
                "action_mask": gymnasium.spaces.Box(low=0, high=1, shape=(n_possible_actions,), 
                                          dtype=np.int8
                ),
            }
        )
        for i in self.agents
        }
        
        
    
    def action_space(self, agent):
        return self.action_spaces[agent]
        
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def delete_files_except_last(self, folder_path, file_prefix):
        files = glob.glob(os.path.join(folder_path, f"{file_prefix}*"))
        if len(files) > 2:
            try:
                files_to_keep = max(files, key=os.path.getmtime)
                for file_path in files:
                    if file_path != files_to_keep:
                        try:
                            os.remove(file_path)
                        except (PermissionError, FileNotFoundError) as e:
                            print(f"Errore di permesso: {e}. Impossibile eliminare il file {file_path}.")
            except (PermissionError, FileNotFoundError) as e:
                print(f"Errore di permesso o di file non trovato.")


    def observe(self, agent):
        observation = np.array([self.obs.CGM], dtype=np.float32)
        # legal_moves = self._legal_moves()[agent]
        legal_moves = self._legal_moves(agent) if agent == self.agent_selection else []
        # action_mask = np.array([legal_moves], dtype=np.float32)
        action_mask = np.zeros(n_possible_actions, 'int8')
        for i in legal_moves:
            action_mask[i] = 1
        return {"observation": observation, "action_mask": action_mask}
    
    # def _combine_observations(self, observations):
    #     combined_obs = []
    #     for agent in self.agents:
    #         combined_obs.extend(observations[agent]["observation"])
    #     return np.array(combined_obs)

    # def _legal_moves(self):
    #     
    #     agent_moves = {'Jerry': None, 'Morty': None, 'Rick': None}
        
    #     if self.ipo_s > self.obs.CGM:
    #         agent_moves['Jerry'] = True
    #         agent_moves['Morty'] = False
    #         agent_moves['Rick'] = False
    #     elif self.ipo_s <= self.obs.CGM < self.iper_s:
    #         agent_moves['Jerry'] = False
    #         agent_moves['Morty'] = True
    #         agent_moves['Rick'] = False
    #     elif self.obs.CGM >= self.iper_s:
    #         agent_moves['Jerry'] = False
    #         agent_moves['Morty'] = False
    #         agent_moves['Rick'] = True
        
    #     return agent_moves
    
    def _legal_moves(self, agent):
        
        if self.agent_selection == 'Jerry':
            return [0]
        elif self.agent_selection == 'Morty':
            return[0,1,2,3,4]
        elif self.agent_selection == 'Rick':
            return[5,6,7,8,9]
    
    # def valid_action_mask(self) -> np.ndarray:
    #     # action_mask = np.zeros(n_possible_actions, dtype=np.float32)
    #     action_mask = np.ones(n_possible_actions, "int8")

    #     for agent in self.agents:
    #         # legal_moves = self._legal_moves()[agent]
    #         legal_moves = self._legal_moves(agent)
    #         action_mask[int(legal_moves)] = 1.0  # Assuming legal_moves is the valid action index
    #     return action_mask
    

    def reset(self, seed=None, options=None):
        self.obs, _, _, _ = self.env._raw_reset()
        
        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        
        # self.action_mask = self._legal_moves()
        
        if self.ipo_s > self.obs.CGM:
            self.agent_selection = 'Jerry'

        elif self.ipo_s <= self.obs.CGM < self.iper_s:
            self.agent_selection = 'Morty'
            
        elif self.obs.CGM >= self.iper_s:
            self.agent_selection = 'Rick'
            
        self.action_mask = self._legal_moves(self.agent_selection)

        
        # self.observations = {agent: self.observe(agent) for agent in self.agents}
        
        # combined_obs = {agent: {"observation": self.observations[agent]["observation"], "action_mask": self.action_mask[agent]} for agent in self.agents}
        # combined_obs = {agent: {"observation": self.observations[agent]["observation"], "action_mask": self._legal_moves(agent)} for agent in self.agents}

        # return combined_obs[self.agent_selection], self.infos

    def step(self, action):
        if self.env.training:
            current_time = datetime.now()
            time_suffix = current_time.strftime("%Y%m%d_%H%M%S")
            self.delete_files_except_last(os.path.join(self.folder, self.subfolder_train), 'risultati_training_')
        
        if self.ipo_s > self.obs.CGM:
            self.agent_selection = 'Jerry'

        elif self.ipo_s <= self.obs.CGM < self.iper_s:
            self.agent_selection = 'Morty'
            
        elif self.obs.CGM >= self.iper_s:
            self.agent_selection = 'Rick'

        
        # if self.agent_selection is None:
        #     self.agent_selection = self.agents[0]  # Initialize agent_selection if not already set

        if self.truncations[self.agent_selection] or self.terminations[self.agent_selection]:
            return self._was_dead_step(action)

        mini_action = action / 100
        # mini_action = 0.05
        self.obs, reward, done, truncated, self.infos[self.agent_selection] = self.env._step(mini_action)
        self.rewards[self.agent_selection] = float(reward)  # Convert reward to float
        
        self.terminations[self.agent_selection] = done
        self.truncations[self.agent_selection] = truncated
        
        self.observations = {agent: self.observe(agent) for agent in self.agents}
        self._accumulate_rewards()
        
        CGM = round(self.obs.CGM,3)

        if self.env.training:
            data_diz_temp = {
                'step': int(self.step_num),
                'CGM': CGM,
                'Action': round(mini_action, 3),
                'Active agent': self.agent_selection,
                'Jerry_Reward': round(self.rewards['Jerry'], 3),
                'Morty_Reward': round(self.rewards['Morty'], 3),
                'Rick_Reward': round(self.rewards['Rick'], 3),
                'Total Reward': round(sum(self.rewards.values()), 3),
                'Jerry_Termination': self.terminations['Jerry'],
                'Morty_Termination': self.terminations['Morty'],
                'Rick_Termination': self.terminations['Rick'],
                'Jerry_Truncation': self.truncations['Jerry'],
                'Morty_Truncation': self.truncations['Morty'],
                'Rick_Truncation': self.truncations['Rick'],
                'Infos': None,  # Ensure Infos is handled correctly
            }

            new_df = pd.DataFrame([data_diz_temp])
            self.df_training = pd.concat([self.df_training, new_df])
            self.df_training.to_csv(os.path.join(self.folder, self.subfolder_train, f'risultati_training_{self.env.patient_name}_{self.time_suffix_Min}.csv'), index=False)
            self.step_num += 1

        # combined_obs = {agent: {"observation": self.observations[agent]["observation"], "action_mask": self.action_mask[agent]} for agent in self.agents}
        # combined_obs = {agent: {"observation": self.observations[agent]["observation"], "action_mask": self._legal_moves(agent)} for agent in self.agents}

        # return combined_obs, sum(self.rewards.values()), self.terminations, self.truncations, self.infos
            
    def render(self):
        if self.render_mode == "human":
            self.env.render()
            
    def show_history(self):
        self.env.show_history()

    def close(self):
        self.env.close()
