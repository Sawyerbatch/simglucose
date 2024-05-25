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
    



def eval_action_mask(env_fn, paziente, scenarios, tir_mean_dict, time_suffix, folder_test,
                     num_games=100, num_timesteps=1000,
                     render_mode=None, last_models = False, **env_kwargs):
    
            
    if last_models:
        for m in os.listdir(os.path.join(main_folder, 'Training_'+p)):
            if m.startswith('T1DSimGymnasiumEnv_MARL_'+paziente):
                model = MaskablePPO.load(os.path.join(main_folder, m.split('.')[0]))
                
    else:
        for m in os.listdir('Models'):
            if m.startswith('T1DSimGymnasiumEnv_MARL_'+paziente):
                model = MaskablePPO.load('Models\\'+m.split('.')[0])
    
    print(model)
    
    # model = MaskablePPO.load(latest_policy)

    
    round_rewards = []
    
    tir_dict = {'ripetizione':[],
                'death hypo':[],
                'ultra hypo':[],
                'heavy hypo':[],
                'severe hypo':[],
                'hypo':[],
                'time in range':[],
                'hyper':[],                 
                'severe hyper':[],
                'heavy hyper':[],                    
                'ultra hyper':[], 
                'death hyper':[],
                'HBGI mean':[],
                'HBGI std':[],
                'LBGI mean':[],
                'LBGI std':[],
                'RI mean':[],
                'RI std':[], 
                'test timesteps':[],
                'scenario':[],
                }
    
    with pd.ExcelWriter(os.path.join(folder_test, f'hystory_{paziente}_{time_suffix_Min}.xlsx')) as patient_writer:

        for i, scen in zip(range(1,num_games+1), scenarios.values()):
            # print(scen)
            lista_BG = []
            
            sheet_name = f'Game_{i}'
            print(sheet_name)
            
            scen = [tuple(x) for x in scen]
            test_scenario = CustomScenario(start_time=start_time, scenario=scen)
            
            # env = T1DSimGymnasiumEnv_MARL(
            #     patient_name=paziente,
            #     custom_scenario=test_scenario,
            #     reward_fun=new_reward,
            #     # seed=123,
            #     render_mode="human",
            #     training = False
            #     # n_steps=n_steps
            # )
            
            def create_env(**kwargs):
                env = T1DSimGymnasiumEnv_MARL(**kwargs)
                # Additional wrappers can be added here if needed
                return env

            env_fn = lambda **kwargs: create_env(
                patient_name=paziente,
                custom_scenario=test_scenario,
                reward_fun=new_reward,
                # seed=123,
                render_mode="human",
                training = False
                # n_steps=n_steps
        )
            
            
            env = env_fn(render_mode=render_mode, **env_kwargs)
            
            env.reset(seed=i)
            env.action_space(env.possible_agents[0]).seed(i)
    
            timestep = 0
            
            scores = {agent: 0 for agent in env.possible_agents}
            total_rewards = {agent: 0 for agent in env.possible_agents}
            
            obs = env.reset()  # Resetta l'ambiente e ottieni l'osservazione iniziale
            # print(obs)
            
            
            data_list = []
            
            # cgm_list = list()
            counter_death_hypo = 0
            counter_under_30 = 0
            counter_under_40 = 0
            counter_under_50 = 0
            counter_under_70 = 0
            counter_euglycem = 0
            counter_over_180 = 0
            counter_over_250 = 0
            counter_over_400 = 0
            counter_over_500 = 0
            counter_death_hyper = 0
            
            counter_total = 0
            
            tir = np.zeros(shape=(11,))
            
            while timestep < num_timesteps:
                print(timestep)
                for agent in env.agent_iter():
                    obs, reward, termination, truncation, infos = env.last()
                    observation, action_mask = obs.values()
                    
                    
                    print('observation', observation)
                    print('info:', infos)
                    
                    if observation <= 21:
                        counter_death_hypo += 1
                    elif 21 < observation < 30:
                        counter_under_30 += 1
                    elif 30 <= observation < 40:
                        counter_under_40 += 1
                    elif 40 <= observation < 50:
                        counter_under_50 += 1
                    elif 50 <= observation< 70:
                        counter_under_70 += 1
                    elif 70 <= observation <= 180:
                        counter_euglycem += 1
                    elif 180 < observation <= 250:
                        counter_over_180 += 1
                    elif 250 < observation <= 400:
                        counter_over_250 += 1
                    elif 400 < observation <= 500:
                        counter_over_400 += 1
                    elif 500 < observation < 595:
                        counter_over_500 += 1
                    elif observation >= 595:
                        counter_death_hyper += 1
                        
                    counter_total += 1
                    # print(paziente)
                    # print(i+1)
                    tir[0] = (counter_death_hypo/counter_total)*100
                    print('death_hypo:',tir[0])
                    tir[1] = (counter_under_30 / counter_total) * 100
                    print('under_30:', tir[1])               
                    tir[2] = (counter_under_40 / counter_total) * 100
                    print('under_40:', tir[2])            
                    tir[3] = (counter_under_50 / counter_total) * 100
                    print('under_50:', tir[3])            
                    tir[4] = (counter_under_70 / counter_total) * 100
                    print('under_70:', tir[4])              
                    tir[5] = (counter_euglycem / counter_total) * 100
                    print('euglycem:', tir[5])              
                    tir[6] = (counter_over_180 / counter_total) * 100
                    print('over_180:', tir[6])             
                    tir[7] = (counter_over_250 / counter_total) * 100
                    print('over_250:', tir[7])                
                    tir[8] = (counter_over_400 / counter_total) * 100
                    print('over_400:', tir[8])               
                    tir[9] = (counter_over_500 / counter_total) * 100
                    print('over_500:', tir[9])            
                    tir[10] = (counter_death_hyper / counter_total) * 100
                    print('death_hyper:', tir[10])
                    
                    lista_BG.append(observation)
                    
                    data_list.append({
                            'Timestep': timestep,
                            'CGM': round(env.obs.CGM, 3),
                            # 'BG': infos['bg'],
                            # 'LBGI': infos['lbgi'],
                            # 'HBGI': infos['hbgi'],
                            # 'RISK': infos['risk'],
                            # 'INS': infos['insulin'],
                            # 'Rick_Obs': str(obs['Rick']),
                            # 'Rick_Action': str(round(actions['Rick'][0],3)),
                            'Jerry_Reward': str(round(total_rewards['Jerry'],3)),
                            'Morty_Reward': str(round(total_rewards['Morty'],3)),
                            'Rick_Reward': str(round(total_rewards['Rick'],3)),
                            # 'Morty_Obs': str(obs['Morty']),
                            # 'Morty_Action': str(round(actions['Morty'][0],3)),     
                            # 'Rick_Done': dones['Rick'],
                            # 'Morty_Done': dones['Morty'],
                            # 'Rick_Trunc': truncs['Rick'],
                            # 'Morty_Trunc': truncs['Morty'],
                            # 'Obs': str(obs),
                        })

                    
                    df = pd.DataFrame(data_list)
                    # print(df['Rick_Reward'])
                    
                    # env.close()
                        
                    df.to_excel(patient_writer, sheet_name=sheet_name, index=False)

                    
                    df_hist = env.show_history()
                    
                    
                
                    if termination or truncation:
                        if env.rewards[env.possible_agents[0]] != env.rewards[env.possible_agents[1]]:
                            winner = max(env.rewards, key=env.rewards.get)
                            scores[winner] += env.rewards[winner]
                        for a in env.possible_agents:
                            total_rewards[a] += env.rewards[a]
                        round_rewards.append(env.rewards)
                        break
                    else:
                        if agent == env.possible_agents[0]:
                            act = env.action_space(agent).sample(action_mask)
                        else:
                            act = int(model.predict(observation, action_masks=action_mask, deterministic=True)[0])
                        env.step(act)
                    timestep += 1
                    if timestep >= num_timesteps:
                        break
                if timestep >= num_timesteps:
                    break
                
        
    
            with pd.ExcelWriter(os.path.join(folder_test, f'results_{paziente}_{time_suffix_Min}.xlsx')) as middle_writer:  
            
                LBGI_mean, HBGI_mean, RI_mean, LBGI_std, HBGI_std, RI_std = risk_index_mod(lista_BG, len(lista_BG))
        
                avg_reward = sum(total_rewards.values()) / len(total_rewards.values())
                print("Total rewards:", total_rewards)
                print(f"Average reward: {avg_reward}")
                
                tir_dict['ripetizione'].append(i)
                tir_dict['death hypo'].append(tir[0])
                tir_dict['ultra hypo'].append(tir[1])
                tir_dict['heavy hypo'].append(tir[2])
                tir_dict['severe hypo'].append(tir[3])
                tir_dict['hypo'].append(tir[4])
                tir_dict['time in range'].append(tir[5])
                tir_dict['hyper'].append(tir[6])
                tir_dict['severe hyper'].append(tir[7])
                tir_dict['heavy hyper'].append(tir[8])
                tir_dict['ultra hyper'].append(tir[9])
                tir_dict['death hyper'].append(tir[10])
                tir_dict['HBGI mean'].append(HBGI_mean)
                tir_dict['HBGI std'].append(HBGI_std)
                tir_dict['LBGI mean'].append(LBGI_mean)
                tir_dict['LBGI std'].append(LBGI_std)
                tir_dict['RI mean'].append(RI_mean)
                tir_dict['RI std'].append(RI_std)
                tir_dict['test timesteps'].append(timestep)

                tir_dict['scenario'].append(scen)
                # tir_dict['tempo esecuzione'].append(tempo_impiegato)
                
                df_result = pd.DataFrame(tir_dict)
            
                df_result.to_excel(middle_writer, sheet_name='general_results', index=False)
                # middle_writer.close()
    
                    
        env.close()
        
        return scores, total_rewards, round_rewards



#%%

if __name__ == "__main__":
    
    last_models=False
    # Ottieni la data corrente
    current_time = datetime.now()
    # Formatta la data nel formato desiderato (ad esempio, YYYYMMDD_HHMMSS)
    # day_suffix = current_time.strftime("%Y%m%d")
    time_suffix = current_time.strftime("%Y%m%d_%H%M%S")
    time_suffix_Min = current_time.strftime("%Y%m%d_%H%M")
    # Crea il nome della cartella usando il suffisso di tempo
    # folder_name = f"results_{day_suffix}"
    # folder_name = f"results_{day_suffix}/{time_suffix_Min}"
    # subfolder_test_name = f"test_{time_suffix_Min}"
    general_results_path = f'test_general_results_{time_suffix_Min}.xlsx'
    test_folder = f"Test\\Test_{time_suffix_Min}"
    # Crea la cartella se non esiste già
    os.makedirs(test_folder, exist_ok=True)
    
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
                # 'adult#002',
                # 'adult#003',
                # 'adult#004',
                # 'adult#005',
                # 'adult#006',
                # 'adult#007',
                # 'adult#008',
                # 'adult#009',
                # 'adult#010',
                ]
    
    # test fixed scenario
    with open('scenarios_5_days_1000_times.json') as json_file:
        test_scenarios = json.load(json_file)

    tir_mean_dict = {
                'paziente':[],
                'avg reward':[],
                'death hypo mean':[],
                'death hypo st dev':[],
                'ultra hypo mean':[],
                'ultra hypo st dev':[],
                'heavy hypo mean':[],
                'heavy hypo st dev':[],
                'severe hypo mean':[],
                'severe hypo st dev':[],
                'hypo mean':[],
                'hypo st dev':[],
                'time in range mean':[],
                'time in range st dev':[],
                'hyper mean':[],
                'hyper st dev':[],           
                'severe hyper mean':[],
                'severe hyper st dev':[],
                'heavy hyper mean':[],
                'heavy hyper st dev':[],
                'ultra hyper mean':[],
                'ultra hyper st dev':[],
                'death hyper mean':[],
                'death hyper st dev':[],
                'HBGI mean of means':[],
                'HBGI mean of std':[],
                'LBGI mean of means':[],
                'LBGI mean of std':[],
                'RI mean of means':[],
                'RI mean of std':[],      
                # 'cap iper mean':[],
                # 'cap ipo mean':[],
                # 'soglia iper mean':[],
                # 'soglia ipo mean':[],
                'ripetizioni':[],
                # 'training learning rate':[],
                # 'training n steps':[],
                # 'training timesteps':[],
                # 'test timesteps':[],       
                # 'scenario':[],
                # 'start time': []
                }
        
        
    for p in (pazienti):  
        
        # Crea il nome della cartella usando il suffisso di tempo
        train_folder = os.path.join(main_folder, f"Training_{p}")

        # Crea la cartella se non esiste già
        # os.makedirs(folder, exist_ok=True)
        
        
        
        start_time = datetime.strptime('3/4/2022 12:00 AM', '%m/%d/%Y %I:%M %p')
        
        # train random scenario
        scen_long = create_scenario(n_days)
        train_scenario = CustomScenario(start_time=start_time, scenario=scen_long)#, seed=seed)
        
        
        
        def env(**kwargs):
            env = T1DSimGymnasiumEnv_MARL(**kwargs)
            # env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
            # env = wrappers.AssertOutOfBoundsWrapper(env)
            # env = wrappers.OrderEnforcingWrapper(env)
            return env
        
        
        env_kwargs = {}
        

        # TRAIN
        # env_fn = env(
        #         patient_name=p,
        #         custom_scenario=train_scenario,
        #         reward_fun=new_reward,
        #         # seed=123,
        #         render_mode="human",
        #         training = True,
        #         folder=folder
        #     )
        # train_action_mask(env_fn, train_folder, p, steps=train_timesteps, seed=0, **env_kwargs)
        # last_models = True
        
        
        env_fn = env(
                patient_name=p,
                custom_scenario=train_scenario,
                reward_fun=new_reward,
                # seed=123,
                render_mode="human",
                training = False,
                folder=test_folder
            )
        
        scores, total_rewards, round_rewards = eval_action_mask(env_fn, p,
                                                                test_scenarios,
                                                                tir_mean_dict,
                                                                time_suffix,
                                                                test_folder,
                                                                num_games=10, 
                                                                num_timesteps=1000,
                                                                last_models=last_models,
                                                                render_mode=None,                            
                                                                **env_kwargs)
        
        # Note that use of masks is manual and optional outside of learning,
        # so masking can be "removed" at testing time
        # model.predict(observation, action_masks=valid_action_array)
