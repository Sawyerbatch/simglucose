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

        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv_MARL(patient, sensor, pump, scenario)
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
        self.action_spaces = gymnasium.spaces.Dict({i: gymnasium.spaces.Box(0, self.env.max_basal, 
                                                    shape=(1,)) for i in self.agents})
        # self.action_space = ParallelEnv.spaces.Dict({i: ParallelEnv.spaces.Box(0, self.env.max_basal, 
        #                                             shape=(1,)) for i in self.agents})
        
        
        
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
                        low=0, high=self.env.max_basal, shape=(100,) #dtype=np.float32
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
        
    # def step(self, action):
    def step(self, actions):
        # obs, reward, done, info = self.env.step(action)
        obs, reward, done, info = self.env.step(actions)
        # Truncated will be controlled by TimeLimit wrapper when registering the env.
        # For example,
        # register(
        #     id="simglucose/adolescent2-v0",
        #     entry_point="simglucose.envs:T1DSimGymnaisumEnv_MARL",
        #     max_episode_steps=10,
        #     kwargs={"patient_name": "adolescent#002"},
        # )
        # Once the max_episode_steps is set, the truncated value will be overridden.
        truncated = False
        # return np.array([obs.CGM], dtype=np.float32), reward, done, truncated, info
        observations = {i:np.array([obs.CGM], dtype=np.float32) for i in self.agents}
        rewards ={i:0 for i in self.agents}
        # rewards ={i:reward for i in self.agents}
        # terminations = {i:done for i in self.agents}
        terminations = {i:done for i in self.agents}
        # truncations = {i:done for i in self.agents}
        truncations = {i:truncated for i in self.agents}
        infos = {i:info for i in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    


    def reset(self, seed=None, options=None):
        # super().reset(seed=seed)
        
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
        
        
        obs, _, _, info = self.env._raw_reset()
        # return np.array([obs.CGM], dtype=np.float32), info
        return {i:np.array([obs.CGM], dtype=np.float32) for i in self.agents}, {i:info for i in self.agents}

    def render(self):
        if self.render_mode == "human":
            self.env.render()

    def close(self):
        self.env.close()
