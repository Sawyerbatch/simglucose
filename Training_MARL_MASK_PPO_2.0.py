# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 23:52:14 2024

@author: Daniele
"""

from __future__ import annotations

import shutil
import scipy
import openpyxl
import csv
import glob
import os
import time
from typing import Optional
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import numpy as np
from pettingzoo.sisl import waterworld_v4
import pandas as pd
import gymnasium
import time
import warnings
import json
from statistics import mean, stdev
from datetime import datetime
from stable_baselines3 import PPO
from simglucose.envs import T1DSimGymnasiumEnv_MARL
from simglucose.simulation.scenario import CustomScenario
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from pettingzoo.utils import wrappers
import pettingzoo
import gymnasium as gym
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO

# Disable all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def new_func(x):
    return -0.0417 * x**2 + 10.4167 * x - 525.0017

def new_reward(BG_last_hour):
    # print('USIAMO LA NOSTRA REWAAAAAAAAAAARD')
    return new_func(BG_last_hour[-1])

def create_scenario(n_days, cho_daily=230):

  scenario = []
  cho_sum = 0
  mu_break, sigma_break = 8, 3 
  mu_lunch, sigma_lunch = 13, 1
  mu_snack, sigma_snack = 17, 2
  mu_dinner, sigma_dinner = 21, 2
  mu_night, sigma_night = 24, 2

  for i in range(n_days):

    mu_cho_break, sigma_cho_break = cho_daily*0.15, 15 
    mu_cho_lunch, sigma_cho_lunch = cho_daily*0.45, 45
    mu_cho_snack, sigma_cho_snack = cho_daily*0.05, 5
    mu_cho_dinner, sigma_cho_dinner = cho_daily*0.35, 35
    mu_cho_night, sigma_cho_night = cho_daily*0.05, 5

    hour_break = int(np.random.normal(mu_break, sigma_break/2)) + 24*i
    hour_lunch = int(np.random.normal(mu_lunch, sigma_lunch/2)) + 24*i
    hour_snack = int(np.random.normal(mu_snack, sigma_snack/2)) + 24*i
    hour_dinner = int(np.random.normal(mu_dinner, sigma_dinner/2)) + 24*i
    hour_night = int(np.random.normal(mu_night, sigma_night/2)) + 24*i

    cho_break = int(np.random.normal(mu_cho_break, sigma_cho_break/2))
    cho_lunch = int(np.random.normal(mu_cho_lunch, sigma_cho_lunch/2))
    cho_snack = int(np.random.normal(mu_cho_snack, sigma_cho_snack/2))
    cho_dinner = int(np.random.normal(mu_cho_dinner, sigma_cho_dinner/2))
    cho_night = int(np.random.normal(mu_cho_night, sigma_cho_night/2))

    if int(np.random.randint(100)) < 60:
      scenario.append((hour_break,cho_break))
    if int(np.random.randint(100)) < 100:
      scenario.append((hour_lunch,cho_lunch))
    if int(np.random.randint(100)) < 30:
      scenario.append((hour_snack,cho_snack))
    if int(np.random.randint(100)) < 95:
      scenario.append((hour_dinner,cho_dinner))
    if int(np.random.randint(100)) < 3:
      scenario.append((hour_night,cho_night))

    #cho_sum += cho_break + cho_lunch + cho_snack + cho_dinner + cho_night

  return scenario

def risk_index_mod(BG, horizon):
    # BG is in mg/dL
    # horizon in samples
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        BG_to_compute = BG[-horizon:]
        fBG = 1.509 * (np.log(BG_to_compute)**1.084 - 5.381)
        rl = 10 * fBG[fBG < 0]**2
        rh = 10 * fBG[fBG > 0]**2
        LBGI_mean = np.nan_to_num(np.mean(rl))
        HBGI_mean = np.nan_to_num(np.mean(rh))
        RI_mean = LBGI_mean + HBGI_mean
        LBGI_std = np.nan_to_num(np.std(rl))
        HBGI_std = np.nan_to_num(np.std(rh))
        RI_std = np.sqrt(LBGI_std**2 + HBGI_std**2)  # Propagation of uncertainty
    return (LBGI_mean, HBGI_mean, RI_mean, LBGI_std, HBGI_std, RI_std)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def mean_std(valori):
    n = len(valori)
    if n == 0:
        return 0.0

def training_supersuit(
    env_fn, paziente, folder, n_steps: int = 2400, steps: int = 2400,
    seed: int | None = 0, **env_kwargs
):
    # Train a single model to play as each agent in a cooperative Parallel environment
    # env = env_fn.parallel_env(**env_kwargs)
    
    env = env_fn

    env.reset(seed=seed)

    print(f"Starting training on {paziente} with {str(env.metadata['name'])}.")

    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 2, num_cpus=6, base_class="stable_baselines3")

    # Note: Waterworld's observation space is discrete (242,) so we use an MLP policy rather than CNN
    # model = PPO(
    #     MlpPolicy,
    #     env,
    #     gamma=0.99,
    #     verbose=3,
    #     learning_rate=1e-3,
    #     batch_size=64,
    #     # batch_size=32,
    #     n_epochs= 10,
    #     n_steps= n_steps
    # )
    
    model = MaskablePPO("MlpPolicy", env, gamma=0.4, seed=32, verbose=1)

    model.learn(total_timesteps=steps, progress_bar=True, reset_num_timesteps=False)

    # model_folder = f"Training\Training_{time_suffix_Min}"
    model.save(os.path.join(folder, f"{env.unwrapped.metadata.get('name')}_{paziente}_{n_steps}_{time_suffix_Min}"))
    
    
    print("Model has been saved.")

    print(f"Finished training on {paziente} with {str(env.unwrapped.metadata['name'])}.")

    env.close()
    
    return model



class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper):
    """Wrapper to allow PettingZoo environments to be used with SB3 illegal action masking."""

    def reset(self, seed=None, options=None):
        """Gymnasium-like reset function which assigns obs/action spaces to be the same for each agent.

        This is required as SB3 is designed for single-agent RL and doesn't expect obs/action spaces to be functions
        """
        super().reset(seed, options)

        # Strip the action mask out from the observation space
        self.observation_space = super().observation_space(self.possible_agents[0])[
            "observation"
        ]
        self.action_space = super().action_space(self.possible_agents[0])

        # Return initial observation, info (PettingZoo AEC envs do not by default)
        return self.observe(self.agent_selection), {}

    def step(self, action):
        """Gymnasium-like step function, returning observation, reward, termination, truncation, info."""
        super().step(action)
        return super().last()

    def observe(self, agent):
        """Return only raw observation, removing action mask."""
        return super().observe(agent)["observation"]

    def action_mask(self):
        """Separate function used in order to access the action mask."""
        return super().observe(self.agent_selection)["action_mask"]

    
    
def mask_fn(env):
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_mask()


def train_action_mask(env_fn, folder, paziente, steps=10_000, seed=0, **env_kwargs):
    """Train a single model to play as each agent in a zero-sum game environment using invalid action masking."""
    # env = env_fn.env(**env_kwargs)

    print(f"Starting training on patient {paziente} with {str(env_fn.metadata['name'])}.")

    # Custom wrapper to convert PettingZoo envs to work with SB3 action masking
    env = SB3ActionMaskWrapper(env_fn)

    env.reset(seed=seed)  # Must call reset() in order to re-define the spaces

    env = ActionMasker(env, mask_fn)  # Wrap to enable masking (SB3 function)
    # MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
    # with ActionMasker. If the wrapper is detected, the masks are automatically
    # retrieved and used when learning. Note that MaskablePPO does not accept
    # a new action_mask_fn kwarg, as it did in an earlier draft.
    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
    model.set_random_seed(seed)
    model.learn(total_timesteps=steps, progress_bar=True, 
                reset_num_timesteps=False)

    # model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")
    model.save(os.path.join(folder, f"{env.unwrapped.metadata.get('name')}_{paziente}_{steps}_{time_suffix_Min}"))
    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()
    

# def env(**kwargs):
#     env = T1DSimGymnasiumEnv_MARL(**kwargs)
#     env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
#     env = wrappers.AssertOutOfBoundsWrapper(env)
#     env = wrappers.OrderEnforcingWrapper(env)
#     return env



#%%

if __name__ == "__main__":
    
    # Ottieni la data corrente
    current_time = datetime.now()
    # Formatta la data nel formato desiderato (ad esempio, YYYYMMDD_HHMMSS)
    # day_suffix = current_time.strftime("%Y%m%d")
    time_suffix = current_time.strftime("%Y%m%d_%H%M%S")
    time_suffix_Min = current_time.strftime("%Y%m%d_%H%M")
    
    main_folder = os.path.join("Training", f"Training_{time_suffix_Min}")
    
    n_days = 5
    # train_timesteps = 2400
    # train_timesteps = 100
    # train_timesteps = 1024
    train_timesteps = 2048
    # n_steps=2
    # n_steps = 1024
    # n_steps = 2048
    # n_steps = 100
    
    
    pazienti = [
                'adult#001',
                'adult#002',
                'adult#003',
                'adult#004',
                'adult#005',
                'adult#006',
                'adult#007',
                'adult#008',
                'adult#009',
                'adult#010',
                ]
    
    # test fixed scenario
    with open('scenarios_5_days_1000_times.json') as json_file:
        test_scenarios = json.load(json_file)
        
        
    for p in (pazienti):  
        
        # Crea il nome della cartella usando il suffisso di tempo
        folder = os.path.join(main_folder, f"Training_{p}")

        # Crea la cartella se non esiste già
        # os.makedirs(folder, exist_ok=True)
        
        
        
        start_time = datetime.strptime('3/4/2022 12:00 AM', '%m/%d/%Y %I:%M %p')
        
        # train random scenario
        scen_long = create_scenario(n_days)
        train_scenario = CustomScenario(start_time=start_time, scenario=scen_long)#, seed=seed)
        
        
        
        def env(**kwargs):
            env = T1DSimGymnasiumEnv_MARL(**kwargs)
            env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
            env = wrappers.AssertOutOfBoundsWrapper(env)
            env = wrappers.OrderEnforcingWrapper(env)
            return env
        
        
        env_kwargs = {}
        
        env_fn = env(
                patient_name=p,
                custom_scenario=train_scenario,
                reward_fun=new_reward,
                # seed=123,
                render_mode="human",
                training = True,
                folder=folder
            )

        # Train a model against itself (takes ~20 seconds on a laptop CPU)
        train_action_mask(env_fn, folder, p, steps=train_timesteps, seed=0, **env_kwargs)
    

        
        # Note that use of masks is manual and optional outside of learning,
        # so masking can be "removed" at testing time
        # model.predict(observation, action_masks=valid_action_array)
