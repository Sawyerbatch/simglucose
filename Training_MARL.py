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
    env_fn, paziente, n_steps: int = 2400, steps: int = 2400,
    seed: int | None = 0, **env_kwargs
):
    # Train a single model to play as each agent in a cooperative Parallel environment
    # env = env_fn.parallel_env(**env_kwargs)
    
    env = env_fn

    env.reset(seed=seed)

    print(f"Starting training on {paziente} with {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=6, base_class="stable_baselines3")

    # Note: Waterworld's observation space is discrete (242,) so we use an MLP policy rather than CNN
    model = PPO(
        MlpPolicy,
        env,
        gamma=0.99,
        verbose=3,
        learning_rate=1e-3,
        batch_size=64,
        # batch_size=32,
        n_epochs= 10,
        n_steps= n_steps
    )

    model.learn(total_timesteps=steps, progress_bar=True, reset_num_timesteps=False)

    model_folder = f"Training\Training_{time_suffix_Min}"
    model.save(os.path.join(model_folder, f"{env.unwrapped.metadata.get('name')}_{paziente}_{n_steps}_{time_suffix_Min}"))
    
    
    print("Model has been saved.")

    print(f"Finished training on {paziente} with {str(env.unwrapped.metadata['name'])}.")

    env.close()
    
    return model


#%%

if __name__ == "__main__":
    
    # Ottieni la data corrente
    current_time = datetime.now()
    # Formatta la data nel formato desiderato (ad esempio, YYYYMMDD_HHMMSS)
    day_suffix = current_time.strftime("%Y%m%d")
    time_suffix = current_time.strftime("%Y%m%d_%H%M%S")
    time_suffix_Min = current_time.strftime("%Y%m%d_%H%M")
    
    folder = f"Training\\Training_{time_suffix_Min}"
    
    n_days = 5
    # train_timesteps = 2400
    # train_timesteps = 100
    # train_timesteps = 1024
    train_timesteps = 1024
    # n_steps=2
    # n_steps = 1024
    n_steps = 2048
    
    
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
        folder += f"\\Training_{p}_{time_suffix_Min}"

        # Crea la cartella se non esiste già
        # os.makedirs(folder, exist_ok=True)
        
        env_kwargs = {}
        
        start_time = datetime.strptime('3/4/2022 12:00 AM', '%m/%d/%Y %I:%M %p')
        
        # train random scenario
        scen_long = create_scenario(n_days)
        train_scenario = CustomScenario(start_time=start_time, scenario=scen_long)#, seed=seed)
        
        env_fn = T1DSimGymnasiumEnv_MARL(
            patient_name=p,
            custom_scenario=train_scenario,
            reward_fun=new_reward,
            # seed=123,
            render_mode="human",
            training = True,
            folder=folder
            # n_steps=n_steps
        )
             
        # Train a model (takes ~3 minutes on GPU)
        trained_model = training_supersuit(env_fn, p, steps=train_timesteps, n_steps=n_steps, seed=42, **env_kwargs)
        