# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:49:08 2024

@author: Daniele
"""

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
from gym.utils import seeding
import gymnasium
from copy import copy, deepcopy
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo.utils import agent_selector
from pettingzoo import ParallelEnv
from typing import Any, Dict
from datetime import datetime


PATIENT_PARA_FILE = pkg_resources.resource_filename(
    "simglucose", "params/vpatient_params.csv"
)

class T1DSimEnv_MARL(gym.Env):
    metadata = {"render_modes": ["human"],
                'name': 'T1DSimGymnasiumEnv_MARL'
                }

    SENSOR_HARDWARE = "Dexcom"
    INSULIN_PUMP_HARDWARE = "Insulet"

    def __init__(
        self, patient_name=None, custom_scenario=None, reward_fun=None, seed=None
    ):
        if patient_name is None:
            patient_name = ["adolescent#001"]

        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.custom_scenario = custom_scenario
        self.env, _, _, _ = self._create_env()

    def _step(self, action: float):
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
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.env = T1DSimEnv_MARL(
            patient_name=patient_name,
            custom_scenario=custom_scenario,
            reward_fun=reward_fun,
            seed=seed,
        )
        
        self.agents = ["Rick", "Morty"]
        self.possible_agents = self.agents[:]
        self.communication_channel = {"Rick": None, "Morty": None}
        
        self.action_spaces = gymnasium.spaces.Dict({i: gymnasium.spaces.Box(0, 0.1, shape=(1,)) for i in self.agents})
        
        self.observation_spaces = gymnasium.spaces.Dict({i: gymnasium.spaces.Dict(
                {
                    "observation": gymnasium.spaces.Box(
                        low=0, high=self.MAX_BG, shape=(1,) #dtype=np.float32
                    ),
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

    def step(self, actions):
        rick_action = np.array([[actions["Rick"]]])
        morty_action = np.array([[actions["Morty"]]])

        rick_obs, rick_reward, rick_done, rick_truncated, rick_info = self.env.step(rick_action)
        morty_obs, morty_reward, morty_done, morty_truncated, morty_info = self.env.step(morty_action)

        current_communication = self.communication_channel.copy()
        current_communication["Morty"] = rick_obs.CGM
        current_communication["Rick"] = morty_obs.CGM

        observations = {
            'Rick': {
                "observation": np.array([rick_obs.CGM], dtype=np.float32),
                "action_mask": np.array([rick_obs.CGM], dtype=np.float32),
                "communication_channel": current_communication
            },
            'Morty': {
                "observation": np.array([morty_obs.CGM], dtype=np.float32),
                "action_mask": np.array([morty_obs.CGM], dtype=np.float32),
                "communication_channel": current_communication
            }
        }

        rewards = {'Rick': rick_reward, 'Morty': morty_reward}
        terminations = {'Rick': rick_done, 'Morty': morty_done}
        truncations = {'Rick': rick_truncated, 'Morty': morty_truncated}
        infos = {'Rick': rick_info, 'Morty': morty_info}

        return observations, rewards, terminations, truncations, infos

    def reset(self, seed=None, options=None):
        self.rick_obs, _, _, _ = self.env._raw_reset()
        self.morty_obs = copy.deepcopy(self.rick_obs)

        self.communication_channel = {"Rick": self.rick_obs.CGM, "Morty": self.morty_obs.CGM}

        observations = {
            'Rick': {
                "observation": np.array([self.rick_obs.CGM], dtype=np.float32),
                "action_mask": np.array([0], dtype=np.float32),
                "communication_channel": self.communication_channel
            },
            'Morty': {
                "observation": np.array([self.morty_obs.CGM], dtype=np.float32),
                "action_mask": np.array([0], dtype=np.float32),
                "communication_channel": self.communication_channel
            }
        }

        rewards = {'Rick': 0.0, 'Morty': 0.0}
        terminations = {'Rick': False, 'Morty': False}
        truncations = {'Rick': False, 'Morty': False}
        infos = {'Rick': {}, 'Morty': {}}

        return observations, infos

    def observe(self, agent):
        return self.env.observe(agent)

    def close(self):
        self.env.close()