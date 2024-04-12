from simglucose.simulation.env import T1DSimEnv_MARL as _T1DSimEnv_MARL
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action
import numpy as np
import pkg_resources
import gym
from gym import spaces
# from gymnasium import spaces
from gym.utils import seeding
from datetime import datetime
import gymnasium
from copy import copy
from gymnasium.spaces import Discrete, MultiDiscrete
# MARL-pettingzoo
# https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/classic/tictactoe/tictactoe.py
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo import ParallelEnv

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
        self, patient_name=None, custom_scenario=None, reward_fun=None, seed=None
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

        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv_MARL(patient, sensor, pump, scenario, reward_fun)
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
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.env = T1DSimEnv_MARL(
            patient_name=patient_name,
            custom_scenario=custom_scenario,
            reward_fun=reward_fun,
            seed=seed,
            # n_steps=n_steps,
        )
        
        
        self.agents = ["Rick", "Morty"]
        self.possible_agents = self.agents[:]
        
        self.communication_channel = {"Rick": None, "Morty": None}
        
        # self.rick_obs = None
        # self.rick_reward = 0.0
        # self.rick_done = False
        # self.rick_info = {}
        
        # self.morty_obs = None
        # self.morty_reward = 0.0
        # self.morty_done = False
        # self.morty_info = {}
        
        # self.action_space = {
        #     i: gymnasium.spaces.Box(
        #         low=0, high=self.env.max_basal, shape=(1,), dtype=np.float32
        #         )
        #     for i in self.agents
        #     }
        
        # self.action_space = {i: spaces.Dict({'action':spaces.Discrete(100)}) for i in self.agents}
        # self.action_space = {i: spaces.Discrete(100) for i in self.agents}
        
        # self.action_space = gymnasium.spaces.Space({i: gymnasium.spaces.Box(0, self.env.max_basal, 
        #                                             shape=(1,)) for i in self.agents})
        # self.action_spaces = gymnasium.spaces.Dict({i: gymnasium.spaces.Box(0, self.env.max_basal, 
        #                                             shape=(1,)) for i in self.agents})
        # self.action_space = ParallelEnv.spaces.Dict({i: ParallelEnv.spaces.Box(0, self.env.max_basal, 
        #                                             shape=(1,)) for i in self.agents})
        
        self.action_spaces = gymnasium.spaces.Dict({i: gymnasium.spaces.Box(0, 0.1, 
                                                    shape=(1,)) for i in self.agents})
        
        
         # self.action_space = spaces.Tuple(
    # [spaces.Dict({'action': spaces.Discrete(100)}) for _ in range(len(self.agents))])
        
        # self.action_space = spaces.Dict({
        #     i: spaces.Discrete(100) for i in self.agents
        # })
        
        
        # self.observation_space = gymnasium.spaces.Dict({i: gymnasium.spaces.Box(
        #     low=0, high=self.MAX_BG, shape=(1,), dtype=np.float32) for i in self.agents})
        
        # self.observation_space = gymnasium.spaces.Space({i: gymnasium.spaces.Box(0, self.env.max_basal, 
        #                                             shape=(1,)) for i in self.agents})
        
        # self.observation_spaces = gymnasium.spaces.Space({i: gymnasium.spaces.Dict(
        #         {
        #             "observation": gymnasium.spaces.Box(
        #                 low=0, high=self.MAX_BG, shape=(1,) #dtype=np.float32
        #             ),
        #             # "action_mask": spaces.Box(
        #             #     low=0, high=self.env.max_basal, shape=(1,), #dtype=np.float32
        #             "action_mask": gymnasium.spaces.Box(
        #                 low=0, high=self.env.max_basal, shape=(100,) #dtype=np.float32
        #             ),
        #         }
        #     )
        #     for i in self.agents})
        
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
    
        # self.observation_spaces = {
        #     a:gymnasium.spaces.Box(low=0, high=self.MAX_BG, shape=(1,), dtype=np.float32)
        #     for a in self.agents}
        
        
        # self.observation_space = ParallelEnv.spaces.Dict({i: ParallelEnv.spaces.Dict(
        #         {
        #             "observation": ParallelEnv.spaces.Box(
        #                 low=0, high=self.MAX_BG, shape=(1,) #dtype=np.float32
        #             ),
        #             # "action_mask": spaces.Box(
        #             #     low=0, high=self.env.max_basal, shape=(1,), #dtype=np.float32
        #             "action_mask": ParallelEnv.spaces.Box(
        #                 low=0, high=self.env.max_basal, shape=(100,) #dtype=np.float32
        #             ),
        #         }
        #     )
        #     for i in self.agents})
        
        
        # self.observation_space = gymnasium.spaces.Box(
        #     low=0, high=self.MAX_BG, shape=(1,), dtype=np.float32
        # )
        
        # self.action_space = gymnasium.spaces.Box(
        #     low=0, high=self.env.max_basal, shape=(1,), dtype=np.float32
        # )
    
    def action_space(self, agent):
        return self.action_spaces[agent]
        
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    
    
    # def observation_space(self, agent):
    #     # return self.observation_spaces[agent]
    #     return MultiDiscrete([7 * 7] * 3)

    # def action_space(self, agent):
    #     print('ciao')
    #     # return self.action_spaces[agent]
    #     return MultiDiscrete([7 * 7] * 3)

        # self.rewards = {i: 0 for i in self.agents}
        # self.terminations = {i: False for i in self.agents}
        # self.truncations = {i: False for i in self.agents}
        # self.infos = {i: {"legal_moves": list(range(0, 9))} for i in self.agents}

        # self._agent_selector = agent_selector(self.agents)
        # self.agent_selection = self._agent_selector.reset()
        
        # self.observation_space = gymnasium.spaces.Box(
        #     low=0, high=self.MAX_BG, shape=(1,), dtype=np.float32
        # )
        
        # self.action_space = gymnasium.spaces.Box(
        #     low=0, high=self.env.max_basal, shape=(1,), dtype=np.float32
        # )
        
    # def step(self, actions):
    #     current_glucose_level = self.obs['CGM']  # Assicurati che self.obs sia un dizionario
        
    #     # Determina quale azione applicare in base al livello di glucosio
    #     if current_glucose_level > 120:
    #         action_to_apply = actions["Rick"]
    #     elif current_glucose_level < 85:
    #         action_to_apply = 0.0  # Assumendo che l'ambiente accetti un'azione scalare
    #     else:
    #         action_to_apply = actions["Morty"]
        
    #     # Supponendo che self.env.step accetti e restituisca il formato corretto per l'azione
    #     obs, reward, done, info = self.env.step(action_to_apply)
        
    #     # Assicurati che il formato di `obs` sia gestibile per il tuo caso d'uso
    #     # Qui si assume che `obs` possa essere direttamente utilizzato per entrambi gli agenti
        
    #     observations = {'Rick': obs, 'Morty': obs}
    #     rewards = {'Rick': reward, 'Morty': reward}
    #     dones = {'Rick': done, 'Morty': done}
    #     truncations = {'Rick': done, 'Morty': done}  # Uguale a `dones` se non distingui tra done e truncation
    #     infos = {'Rick': info, 'Morty': info}
    
    #     return observations, rewards, dones, truncations, infos
    # def step(self, action):
    def step(self, actions):
        
        rick_action = actions["Rick"]
        morty_action = actions["Morty"]
        safe_action = np.array([[0.0]])
        
        # Aggiorna lo stato del canale di comunicazione
        self.communication_channel["Rick"] = self.obs.CGM
        self.communication_channel["Morty"] = self.obs.CGM
        
        iper_s = 120
        ipo_s = 85
        
        # # Logica delle azioni basata sulla CGM
        # if rick_obs.CGM < ipo_s:
        #     rick_action = azione_in_base_a_ipo_s
        # elif rick_obs.CGM > iper_s:
        #     rick_action = azione_in_base_a_iper_s
        
        # if np.array([obs.CGM], dtype=np.float32) < ipo_s:
        #     return np.array([obs.CGM], dtype=np.float32), rick_reward, rick_done, rick_truncated, rick_info
        # elif np.array([obs.CGM], dtype=np.float32) > iper_s:
        #     return np.array([obs.CGM], dtype=np.float32), reward, done, truncated, info
        # else:
        #     return np.array([obs.CGM], dtype=np.float32), reward, done, truncated, info
        
        if self.obs.CGM > iper_s:
            morty_action =  [0.0]
            # self.rick_obs, self.rick_reward, self.rick_done, self.rick_info = self.env.step(rick_action)
            # self.obs, self.rick_reward, self.done, self.info = self.env.step(rick_action)
            self.obs, self.rick_reward, self.done, self.info = self.env.step(rick_action)
            # self.obs, self.rick_reward, self.done, self.info = self.env.step([1.0])

            # self.obs, self.reward, self.done, self.info = self.env.step(rick_action)
            print('Rick action', rick_action)
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
            print('Morty action', morty_action)
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
            
        # rick_obs, rick_reward, rick_done, rick_info = self.env.step(rick_action)
        # print('Rick obs', rick_obs)
        # morty_obs, morty_reward, morty_done, morty_info = self.env.step(morty_action)
        # print('Morty obs', morty_obs)
        
        # self.rick_obs, self.rick_reward, self.rick_done, self.rick_info = self.env.step(rick_action)
        # print('Rick obs', self.rick_obs)
        # self.morty_obs, self.morty_reward, self.morty_done, self.morty_info = self.env.step(morty_action)
        # print('Morty obs', self.morty_obs)
        
        # Aggiorna la comunicazione tra agenti
        # self.communication_channel["Morty"] = self.rick_obs.CGM
        # self.communication_channel["Rick"] = self.morty_obs.CGM
        
        # current_communication = self.communication_channel.copy()
        # obs, reward, done, info = self.env.step(action)
        # obs, reward, done, info = self.env.step(actions)
        
        # Truncated will be controlled by TimeLimit wrapper when registering the env.
        # For example,
        # register(
        #     id="simglucose/adolescent2-v0",
        #     entry_point="simglucose.envs:T1DSimGymnaisumEnv_MARL",
        #     max_episode_steps=10,
        #     kwargs={"patient_name": "adolescent#002"},
        # )
        
        
        # Once the max_episode_steps is set, the truncated value will be overridden.
        # truncated = False
        # self.rick_truncated = False
        # self.morty_truncated = False
        
        
        # return np.array([obs.CGM], dtype=np.float32), reward, done, truncated, info
        # observations = {i:np.array([obs.CGM], dtype=np.float32) for i in self.agents}
        # observations = {'Rick':np.array([self.rick_obs.CGM], dtype=np.float32),
        #                 'Morty':np.array([self.morty_obs.CGM], dtype=np.float32)}
        
        observations = {
            'Rick': {
                "observation": np.array([self.rick_obs.CGM], dtype=np.float32),
                "action_mask": np.array([self.rick_obs.CGM], dtype=np.float32),
                "communication_channel": self.communication_channel
                # "communication_channel": current_communication
            },
            'Morty': {
                "observation": np.array([self.morty_obs.CGM], dtype=np.float32),
                "action_mask": np.array([self.morty_obs.CGM], dtype=np.float32),
                "communication_channel": self.communication_channel
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

    
        return observations, rewards, terminations, truncations, infos
    
    # def reset(self, seed=None, options=None):
    #     self.agents = copy(self.possible_agents)
        
    #     # Reset dell'ambiente sottostante e ottieni l'osservazione iniziale
    #     self.obs, _, _, _ = self.env._raw_reset()  # Aggiorna questo in base alla funzionalità esatta di _raw_reset
        
    #     # Inizializza i valori per "Rick" e "Morty"
    #     self.rick_reward, self.rick_done, self.rick_truncated, self.rick_info = 0.0, False, False, {}
    #     self.morty_reward, self.morty_done, self.morty_truncated, self.morty_info = 0.0, False, False, {}
        
    #     # Assumi che 'self.obs' sia l'osservazione iniziale corretta da utilizzare per entrambi gli agenti
    #     observations = {
    #         'Rick': {"observation": np.array([self.obs.CGM], dtype=np.float32), "action_mask": np.array([0], dtype=np.float32)},
    #         'Morty': {"observation": np.array([self.obs.CGM], dtype=np.float32), "action_mask": np.array([0], dtype=np.float32)}
    #     }
        
    #     infos = {a: {} for a in self.agents}  # Prepara gli 'infos' iniziali per ogni agente
    
    #     return observations, infos

    def reset(self, seed=None, options=None):
        # super().reset(seed=seed)
        self.agents = copy(self.possible_agents)
        # self.observation_spaces = gymnasium.spaces.Space({i: gymnasium.spaces.Dict(
        #         {
        #             "observation": gymnasium.spaces.Box(
        #                 low=0, high=self.MAX_BG, shape=(1,) #dtype=np.float32
        #             ),
        #             # "action_mask": spaces.Box(
        #             #     low=0, high=self.env.max_basal, shape=(1,), #dtype=np.float32
        #             "action_mask": gymnasium.spaces.Box(
        #                 low=0, high=self.env.max_basal, shape=(100,) #dtype=np.float32
        #             ),
        #         }
        #     )
        #     for i in self.agents})
        
        
        
        # obs, _, _, info = self.env._raw_reset()
        
        # observations = {
        #     "Rick": {"observation": obs},#, "action_mask": [0, 1, 1, 0]},
        #     "Morty": {"observation": obs},#, "action_mask": [1, 0, 0, 1]},
        # }
        
        self.rick_reward, self.rick_done, self.rick_truncated, self.rick_info = 0.0, False, False, {}
        self.morty_reward, self.morty_done, self.morty_truncated, self.morty_info = 0.0, False, False, {}
        
        self.obs, _, _, info = self.env._raw_reset()
        
        self.rick_obs = self.obs
        self.morty_obs = self.obs
        
        # observations = {
        #     "Rick": {"observation": self.rick_obs, "action_mask": [0]},
        #     "Morty": {"observation": self.morty_obs, "action_mask": [0]},
        # }
        
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
        
        # observations = {
        #         'Rick': {"observation": np.array([self.obs.CGM], dtype=np.float32), "action_mask": np.array([0], dtype=np.float32)},
        #         'Morty': {"observation": np.array([self.obs.CGM], dtype=np.float32), "action_mask": np.array([0], dtype=np.float32)}
        #     }
            
        
        # print(observations)

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos
        
        # return np.array([obs.CGM], dtype=np.float32), info
        # return {i:np.array([obs.CGM], dtype=np.float32) for i in self.agents}, {i:info for i in self.agents}

    def render(self):
        if self.render_mode == "human":
            self.env.render()
            
    def show_history(self):
        self.env.show_history()

    def close(self):
        self.env.close()