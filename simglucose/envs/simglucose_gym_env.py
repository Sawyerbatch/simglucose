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
# from gymnasium import spaces
from gym.utils import seeding
from datetime import datetime, timedelta
import gymnasium
from copy import copy
from gymnasium.spaces import Discrete, MultiDiscrete
# MARL-pettingzoo
# https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/classic/tictactoe/tictactoe.py
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo import ParallelEnv
import shutil

from typing import Any, Dict, Generic, Iterable, Iterator, TypeVar
ObsType = TypeVar("ObsType")

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    "simglucose", "params/vpatient_params.csv"
)



class T1DSimEnv_MARL(gym.Env):
    metadata = {"render_modes": ["human"],
                    'name': 'T1DSimGymnasiumEnv_MARL'
                    }
    """
    A wrapper of simglucose.simulation.env.T1DSimEnv_MARL to support gym API
    """

    metadata = {"render.modes": ["human"]}

    SENSOR_HARDWARE = "Dexcom"
    INSULIN_PUMP_HARDWARE = "Insulet"

    def __init__(
        self, patient_name=None, custom_scenario=None, reward_fun=None,
        seed=None, training=None, folder=None
    ):
        """
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        """
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            patient_name = ["adolescent#001"]

        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.step_num = 1
        self.training = training
        self.folder = folder
        self.np_random, _ = seeding.np_random(seed=seed)
        self.custom_scenario = custom_scenario
        self.env, _, _, _ = self._create_env()
        
        

    def _step(self, action: float):
        # This gym only controls basal insulin
        act = Action(basal=action, bolus=0)
        if self.reward_fun is None:
            return self.env.step(act)
        return self.env.step(act, reward_fun=self.reward_fun)

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
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31
        seed4 = seeding.hash_seed(seed3 + 1) % 2**31

        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)

        if isinstance(self.patient_name, list):
            patient_name = self.np_random.choice(self.patient_name)
            patient = T1DPatient.withName(patient_name, random_init_bg=True, seed=seed4)
        else:
            patient = T1DPatient.withName(
                self.patient_name, random_init_bg=True, seed=seed4
            )

        if isinstance(self.custom_scenario, list):
            scenario = self.np_random.choice(self.custom_scenario)
            
        else:
            scenario = (
                RandomScenario(start_time=start_time, seed=seed3)
                if self.custom_scenario is None
                else self.custom_scenario
            )
        
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

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1000, shape=(1,))

    @property
    def max_basal(self):
        return self.env.pump._params["max_basal"]


# class T1DSimGymnasiumEnv_MARL(gymnasium.Env):
class T1DSimGymnasiumEnv_MARL(ParallelEnv):
    metadata = {"render_modes": ["human"],
                'name': 'T1DSimGymnasiumEnv_MARL'
                }
    
    MAX_BG = 1000

    def __init__(
        self,
        patient_name=None,
        custom_scenario=None,
        reward_fun=None,
        seed=None,
        render_mode=None,
        # n_steps=None
        training=None,
        folder=None
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.env = T1DSimEnv_MARL(
            patient_name=patient_name,
            custom_scenario=custom_scenario,
            reward_fun=reward_fun,
            seed=seed,
            # n_steps=n_steps,
            training=training,
            folder=folder
        )
        
        if self.env.training == True:
            current_time = datetime.now()
            time_suffix_Min = current_time.strftime("%Y%m%d_%H%M")
            day_suffix = current_time.strftime("%Y%m%d")
            self.folder = folder
            # self.folder = f"Training\Training_{time_suffix_Min}\Training_{self.env.patient_name}_{time_suffix_Min}"
            # self.subfolder_name_train = f"training_{self.env.patient_name}_{time_suffix_Min}"        
            # self.df_training = pd.read_csv('risultati_training.csv')
            os.makedirs(os.path.join(self.folder), exist_ok=True)
            
            data_diz = {
                'step': [],
                'BG': [],
                'Rick_Action': [],
                'Rick_Reward': [],
                'Morty_Action': [],
                'Morty_Reward': [],
                'Rick_Done': [],
                'Morty_Done': [],
                'Rick_Trunc': [],
                'Morty_Trunc': [],
                'Obs': []
            }
        
            # Crea un DataFrame vuoto con le colonne specificate nel dizionario
            self.df_training = pd.DataFrame(data_diz)
            
            
        
        self.agents = ["Rick", "Morty"]
        self.possible_agents = self.agents[:]
        
        self.communication_channel = {"Rick": None, "Morty": None}
        
      
        self.action_spaces = gymnasium.spaces.Dict({i: gymnasium.spaces.Box(0, 0.1, 
                                                    shape=(1,)) for i in self.agents})
        
        
        self.observation_spaces = gymnasium.spaces.Dict({i: gymnasium.spaces.Dict(
                {
                    "observation": gymnasium.spaces.Box(
                        low=0, high=self.MAX_BG, shape=(1,) #dtype=np.float32
                    ),
                    # "action_mask": spaces.Box(
                    #     low=0, high=self.env.max_basal, shape=(1,), #dtype=np.float32
                    "action_mask": gymnasium.spaces.Box(
                        low=0, high=self.env.max_basal, shape=(1,) #dtype=np.float32
                    ),
                    
                }
            )
            for i in self.agents})
    
    
    def action_space(self, agent):
        return self.action_spaces[agent]
        
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    
    def delete_files_except_last(self, folder_path, file_prefix):
        # Costruisci il percorso completo dei file
        files = glob.glob(os.path.join(folder_path, f"{file_prefix}*"))
        
        # Mantieni solo l'ultimo file nella lista
        if len(files) > 2:
            try:
                files_to_keep = max(files, key=os.path.getmtime)
                # Elimina tutti i file tranne l'ultimo
                for file_path in files:
                    if file_path != files_to_keep:
                        try:
                            os.remove(file_path)
                        # except PermissionError as e:
                        # except FileNotFoundError as e:
                        except (PermissionError, FileNotFoundError) as e:
                            print(f"Errore di permesso: {e}. Impossibile eliminare il file {file_path}.")
            except (PermissionError, FileNotFoundError) as e:
                print(f"Errore di permesso o di file non trovato.")

    
    def step(self, actions):

        if self.env.training == True:
            current_time = datetime.now()
            # Formatta la data nel formato desiderato (ad esempio, YYYYMMDD_HHMMSS)
            # day_suffix = current_time.strftime("%Y%m%d")
            time_suffix = current_time.strftime("%Y%m%d_%H%M%S")
            # time_suffix_Min = current_time.strftime("%Y%m%d_%H%M")
            # Sottrai un secondo dall'ora attuale
            # previous_time = current_time - timedelta(seconds=1)
            # folder_name = f"results_{day_suffix}/{time_suffix_Min}"
            # folder_name = f"results_{day_suffix}"
            # subfolder_name = f"{time_suffix_Min1}"
            # Formatta la data e l'ora nel formato desiderato
            # time_suffix_prec = previous_time.strftime("%Y%m%d_%H%M%S")
            
            self.delete_files_except_last(os.path.join(self.folder), 'risultati_training_')


        rick_action = actions["Rick"]
        morty_action = actions["Morty"]
        safe_action = np.array([[0.0]])
        
        # Aggiorna lo stato del canale di comunicazione
        self.communication_channel["Rick"] = self.obs.CGM
        self.communication_channel["Morty"] = self.obs.CGM
        
        iper_s = 120
        ipo_s = 85
        
    
        if self.obs.CGM > iper_s:
            morty_action =  [0.0]
            # self.rick_obs, self.rick_reward, self.rick_done, self.rick_info = self.env.step(rick_action)
            self.obs, self.rick_reward, self.done, self.info = self.env.step(rick_action)

            # self.obs, self.rick_reward, self.done, self.info = self.env.step([1.0])

            # self.obs, self.reward, self.done, self.info = self.env.step(rick_action)
            # print('Rick action', rick_action)
            # print('Morty action', morty_action)
            # self.obs = self.rick_obs
            self.rick_obs = self.obs
            self.morty_obs = self.obs
            self.rick_done = self.done
            self.morty_done = self.done
            self.rick_info = self.info
            self.morty_info = self.info
            print('obs', self.rick_obs)
            # self.communication_channel["Morty"] = self.rick_obs.CGM
        elif ipo_s < self.obs.CGM < iper_s:
            rick_action =  [0.0]
            # self.morty_obs, self.morty_reward, self.morty_done, self.morty_info = self.env.step(morty_action)
            self.obs, self.morty_reward, self.done, self.info = self.env.step(morty_action)
            # self.obs, self.morty_reward, self.done, self.info = self.env.step([[0.08483092]])
            # self.obs, self.reward, self.done, self.info = self.env.step(morty_action)
            # print('Rick action', rick_action)
            # print('Morty action', morty_action)
            # self.obs = self.morty_obs
            self.rick_obs = self.obs
            self.morty_obs = self.obs
            self.rick_done = self.done
            self.morty_done = self.done
            self.rick_info = self.info
            self.morty_info = self.info
            print('obs', self.morty_obs)
            # self.communication_channel["Rick"] = self.morty_obs.CGM
        else:
            rick_action =  [0.0]
            morty_action =  [0.0]
            # self.morty_obs, self.morty_reward, self.morty_done, self.morty_info = self.env.step(safe_action)
            self.obs, _, self.done, self.info = self.env.step(safe_action)
            # self.obs, self.reward, self.done, self.info = self.env.step(safe_action)
            print('Safe action', safe_action)
            self.rick_obs = self.obs
            self.morty_obs = self.obs
            self.rick_done = self.done
            self.morty_done = self.done
            self.rick_info = self.info
            self.morty_info = self.info
            print('obs', self.morty_obs)
            # self.obs = self.morty_obs
            # self.communication_channel["Morty"] = self.morty_obs.CGM
            # self.communication_channel["Rick"] = self.morty_obs.CGM
        
        # print('obs', self.morty_obs)
        
        # actions = {'Rick':rick_action,
        #             'Morty':morty_action}
        # print('actions',actions)
            
        
        observations = {
            'Rick': {
                "observation": np.array([self.rick_obs.CGM], dtype=np.float32),
                "action_mask": np.array([self.rick_obs.CGM], dtype=np.float32),
                # "communication_channel": self.communication_channel
                # "communication_channel": current_communication
            },
            'Morty': {
                "observation": np.array([self.morty_obs.CGM], dtype=np.float32),
                "action_mask": np.array([self.morty_obs.CGM], dtype=np.float32),
                # "communication_channel": self.communication_channel
                # "communication_channel": current_communication
            }
        }
        
        
        # rewards ={i:0 for i in self.agents}
        rewards = {'Rick':self.rick_reward,
                    'Morty':self.morty_reward}
        print(rewards)
        
        # Modifica le condizioni di terminazione
        # max_allowed_bg =  100.0 # 250  
        # min_allowed_bg = 90.0 # 40
        # rick_done = self.rick_obs.CGM > max_allowed_bg or self.rick_obs.CGM < min_allowed_bg
        # morty_done = self.morty_obs.CGM > max_allowed_bg or self.morty_obs.CGM < min_allowed_bg
        # print('!!! rick: ', self.rick_obs, self.rick_done, '; morty:', self.morty_obs, self.morty_done)
        print()
        
        # rewards ={i:reward for i in self.agents}
        # terminations = {i:done for i in self.agents}
        
        # terminations = {i:done for i in self.agents}
        terminations = {'Rick':self.rick_done,
                        'Morty':self.morty_done}
        
        # done = any([rick_done, morty_done])
        
        # truncations = {i:done for i in self.agents}
        # truncations = {i:truncated for i in self.agents}
        truncations = {'Rick':self.rick_truncated,
                        'Morty':self.morty_truncated}
        
        
        # infos = {i:info for i in self.agents}
        infos = {'Rick':self.rick_info,
                  'Morty':self.morty_info}
        # print('INFOOOOO', infos)
        
        
        if self.env.training == True:
            # Dizionario con i dati da inserire nel DataFrame
            data_list = {
                'step': int(self.step_num),
                'BG': self.obs.CGM,
                'Rick_Action': round(rick_action[0],3),
                'Rick_Reward': round(self.rick_reward,3),
                'Morty_Action': round(morty_action[0],3),
                'Morty_Reward': round(self.morty_reward,3),
                'Rick_Done': self.rick_done, 
                'Morty_Done': self.morty_done,
                'Rick_Trunc': self.rick_truncated,
                'Morty_Trunc': self.morty_truncated,
                'Obs': self.obs
            }
            
            # Crea un DataFrame con i nuovi dati
            new_df = pd.DataFrame(data_list)
        
            # Appendi il nuovo DataFrame al DataFrame esistente
            self.df_training = pd.concat([self.df_training, new_df])
            
            # Salva il DataFrame aggiornato in un file CSV
            self.df_training.to_csv(os.path.join(self.folder, f'risultati_training_{self.env.patient_name}_{time_suffix}.csv'), index=False)
            
            self.step_num += 1
        
        
        return observations, rewards, terminations, truncations, infos


    def reset(self, seed=None, options=None):
        # super().reset(seed=seed)
        self.agents = copy(self.possible_agents)
        
        self.infos = {'Rick': {}, 'Morty': {}}
        self.rewards = {'Rick': 0, 'Morty': 0}
        
        self.rick_reward, self.rick_done, self.rick_truncated, self.rick_info = 0.0, False, False, {}
        self.morty_reward, self.morty_done, self.morty_truncated, self.morty_info = 0.0, False, False, {}
        
        self.obs, _, _, info = self.env._raw_reset()
        
        self.rick_obs = self.obs
        self.morty_obs = self.obs
        
        self.step_num = 1
        
        
        observations = {
            'Rick': {
                "observation": np.array([self.rick_obs.CGM], dtype=np.float32),
                "action_mask": np.array([0], dtype=np.float32),
                # "communication_channel": self.communication_channel
            },
            'Morty': {
                "observation": np.array([self.morty_obs.CGM], dtype=np.float32),
                "action_mask": np.array([0], dtype=np.float32),
                # "communication_channel": self.communication_channel
            }
        }
        

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos


    def render(self):
        if self.render_mode == "human":
            self.env.render()
            
    def show_history(self):
        self.env.show_history()

    def close(self):
        self.env.close()