from __future__ import annotations

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:09:12 2024

@author: Daniele
"""

"""Uses Stable-Baselines3 to train agents to play the Waterworld environment using SuperSuit vector envs.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""


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

def train_butterfly_supersuit(
    env_fn, paziente, time_suffix, n_steps: int = 2400,
    seed: int | None = 0, **env_kwargs
):
    # Train a single model to play as each agent in a cooperative Parallel environment
    # env = env_fn.parallel_env(**env_kwargs)
    env = env_fn

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")

    # Note: Waterworld's observation space is discrete (242,) so we use an MLP policy rather than CNN
    model = PPO(
        MlpPolicy,
        env,
        gamma=0.99,
        verbose=3,
        learning_rate=1e-3,
        batch_size=256,
    )

    model.learn(total_timesteps=n_steps, progress_bar=True)

    # model.save(f"model_{paziente}_{time_suffix}")
    
    model.save(f"{env.unwrapped.metadata.get('name')}_{paziente}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()



def evaluation(paziente, model, scenarios, tir_mean_dict, time_suffix, folder_name,
         num_games: int = 100, test_timesteps=10, 
         render_mode: Optional[str] = None, **env_kwargs):
    
    # Initialize the environment
    env = env_fn  # Assicurati che env_fn restituisca un'istanza dell'ambiente

    print(f"\nStarting evaluation on {env.metadata['name']} (num_games={num_games}, render_mode={render_mode})")

    try:
        latest_policy = max(glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime)
    except ValueError:
        print("Policy not found.")
        exit(0)

    if model == None:
        model = PPO.load(latest_policy)
    
    total_rewards = {agent: 0 for agent in env.possible_agents}
    
    # with open('evaluation_results.csv', 'w', newline='') as csvfile:
    #     fieldnames = ['Timestep', 'Obs', 'Morty_Obs', 'Rick_Obs',
    #                   'Morty_Reward', 'Rick_Reward',
    #                   'Morty_Done', 'Rick_Done',
    #                   'Morty_Trunc', 'Rick_Trunc',
    #                   ]
    
        # writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # writer.writeheader()
    
    
        
        
    tir_dict = {'death hypo':[],
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
                'LBGI mean':[],
                'LBGI std':[],
                'HBGI mean':[],
                'HBGI std':[],
                'RI mean':[],
                'RI std':[], 
                # 'cap iper':[],
                # 'cap ipo':[],
                # 'soglia iper':[],
                # 'soglia ipo':[],
                # 'training learning rate':[],
                # 'training n steps':[],
                # 'training timesteps':[],
                'test timesteps':[],
                'ripetizione':[],
                'scenario':[],
                # 'tempo esecuzione':[],
                }
    
    with pd.ExcelWriter(os.path.join(folder_name, f'story_{paziente}_{time_suffix}.xlsx')) as patient_writer:
      
        for game, scen in zip(range(1,num_games+1), scenarios.values()):
            # print(scen)
            lista_BG = []
            
            total_rewards = {agent: 0 for agent in env.possible_agents}
            sheet_name = f'Game_{game}'
            
            # writer.writerow({'Game': f'Game {game}'})
            
            scen = [tuple(x) for x in scen]
            test_scenario = CustomScenario(start_time=start_time, scenario=scen)
            
            env = T1DSimGymnasiumEnv_MARL(
                patient_name=paziente,
                custom_scenario=test_scenario,
                reward_fun=new_reward,
                # seed=123,
                render_mode="human",
                # n_steps=n_steps
            )
            
            
            obs = env.reset()  # Resetta l'ambiente e ottieni l'osservazione iniziale
            # print(obs)
            
            done = False
            
            df = pd.DataFrame(columns=['Timestep', 'Obs', 'Morty_Obs', 'Rick_Obs',
                                        'Morty_Reward', 'Rick_Reward',
                                        'Morty_Done', 'Rick_Done',
                                        'Morty_Trunc', 'Rick_Trunc'])
            
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
            
            for t in range(test_timesteps):
                
                print(f'test n {game}, timestep {t}')
                
                if not done:
                    
                    # print('oooobs', obs)
                    
                    # se obs è una tupla vuol dire che viene dal reset,
                    # e devo scartare un secondo elemento vuoto
                    # se è un dizionario siamo al secondo ciclo e va bene così
                    if type(obs) == tuple:
                        # print('lunghezza obs', len(obs))
                        observ = obs[0]
                    else:
                        # print('lunghezza obs', len(obs), '/n')
                        observ = obs
                        
                    # {'Rick': {'observation': array([142.04321], dtype=float32),
                    # 'action_mask': array([0.], dtype=float32)},
                    # 'Morty': {'observation': array([142.04321], dtype=float32),
                    # 'action_mask': array([0.], dtype=float32)}}
                    actions = {}
         
                    # print(a, type(a))
                    for agent, agent_obs in observ.items():
                        action, _ = model.predict(agent_obs, deterministic=True)  # Ottieni l'azione per ogni agente
                        # esempio agent_obs = 
                        # {'observation': array([142.04321], dtype=float32), 
                        # 'action_mask': array([0.], dtype=float32)}
                        actions[agent] = action
                    
                    obs, rewards, dones, truncs, infos = env.step(actions)  # Esegui un passo dell'ambiente con le azioni degli agenti
                    # {'Rick': {'observation': array([141.73483], dtype=float32),
                    # 'action_mask': array([141.73483], dtype=float32)}, 
                    # 'Morty': {'observation': array([141.73483], dtype=float32),
                    # 'action_mask': array([141.73483], dtype=float32)}}
                    
                    # CON QUESTA IMPLEMENTAZIONE I TRUNC SONO INUTILI
                    # PERCHE' ITERO SUI TIMESTEPS?
                    
                    for agent, reward in rewards.items():
                        total_rewards[agent] += reward  # Aggiorna il totale dei premi
                        
                    observation = obs['Rick']['observation'][0]
                    
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
                    
                    done = all(dones.values())  # Controlla se tutti gli agenti hanno terminato
                    
                    
                    data_list.append({
                            'Timestep': t,
                            'Obs': str(obs),
                            # 'Morty_Obs': str(obs['Morty']),
                            'Morty_Reward': str(rewards['Morty']),
                            # 'Rick_Obs': str(obs['Rick']),
                            'Rick_Reward': str(rewards['Rick']),
                            'Rick_Done': dones['Rick'],
                            'Morty_Done': dones['Morty'],
                            'Rick_Trunc': truncs['Rick'],
                            'Morty_Trunc': truncs['Morty']
                        })
                    
                    df = pd.DataFrame(data_list)
                    # print(df['Rick_Reward'])
                    
                    env.close()
                        
                    df.to_excel(patient_writer, sheet_name=sheet_name, index=False)

            with pd.ExcelWriter(os.path.join(folder_name,f'risultati_finali_{time_suffix}.xlsx')) as final_writer:  
            
                LBGI_mean, HBGI_mean, RI_mean, LBGI_std, HBGI_std, RI_std = risk_index_mod(lista_BG, len(lista_BG))
        
                avg_reward = sum(total_rewards.values()) / len(total_rewards.values())
                print("Total rewards:", total_rewards)
                print(f"Average reward: {avg_reward}")
                
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
                tir_dict['LBGI mean'].append(LBGI_mean)
                tir_dict['LBGI std'].append(LBGI_std)
                tir_dict['HBGI mean'].append(HBGI_mean)
                tir_dict['HBGI std'].append(HBGI_std)
                tir_dict['RI mean'].append(RI_mean)
                tir_dict['RI std'].append(RI_std)
                # iper_mod = insert_dot(iper, 1)
                # tir_dict['cap iper'].append(iper_mod)
                # ipo_mod = insert_dot(ipo, 1)
                # tir_dict['cap ipo'].append(ipo_mod)
                # tir_dict['soglia iper'].append(iper_s)
                # tir_dict['soglia ipo'].append(ipo_s)
                # tir_dict['test timesteps'].append(test_timesteps)
                # tir_dict['game'].append(game+1)
                # tir_dict['training learning rate'].append(training_learning_rate)
                # tir_dict['training n steps'].append(training_n_steps)
                # tir_dict['training timesteps'].append(training_total_timesteps)
                tir_dict['test timesteps'].append(test_timesteps)
                tir_dict['ripetizione'].append(game)
                tir_dict['scenario'].append(scen)
                # tir_dict['tempo esecuzione'].append(tempo_impiegato)
                
                df_result = pd.DataFrame(tir_dict)
            
                df_result.to_excel(final_writer, sheet_name='risultati_'+paziente, index=False)
                # final_writer.save()
    
    return avg_reward, tir_dict

#%%

if __name__ == "__main__":
    
    # Ottieni la data corrente
    current_time = datetime.now()
    
    # Formatta la data nel formato desiderato (ad esempio, YYYYMMDD_HHMMSS)
    day_suffix = current_time.strftime("%Y%m%d")
    time_suffix = current_time.strftime("%Y%m%d_%H%M%S")
    # Crea il nome della cartella usando il suffisso di tempo
    folder_name = f"results_{day_suffix}"
    # Crea la cartella se non esiste già
    os.makedirs(folder_name, exist_ok=True)
    
    n_days = 5
    train_timesteps = 2400
    num_games = 5
    test_timesteps = 100
    
    pazienti = ['adult#001']
    
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
                'LBGI mean of means':[],
                'LBGI mean of std':[],
                'HBGI mean of means':[],
                'HBGI mean of std':[],
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
            # n_steps=n_steps
        )
             
        # Train a model (takes ~3 minutes on GPU)
        # train_butterfly_supersuit(env_fn, p, time_suffix, steps=train_timesteps, seed=42, **env_kwargs)
        
        # se ho già il modello
        # model = PPO.load('T1DSimGymnasiumEnv_MARL_20240320-183729')
        model = PPO.load('T1DSimGymnasiumEnv_MARL_adult#001_20240330-021206')
    
        # test
        avg_reward, tir_dict = evaluation(p, model, test_scenarios, 
                                    tir_mean_dict, time_suffix, folder_name,
                                    num_games=num_games, 
                                    test_timesteps=test_timesteps, 
                                    render_mode=None, **env_kwargs)
        
        with pd.ExcelWriter(os.path.join(folder_name, f'risultati_finali_{time_suffix}.xlsx'), mode='a') as final_writer:
        
            tir_mean_dict['paziente'].append(p)
            tir_mean_dict['avg reward'].append(avg_reward)
            tir_mean_dict['death hypo mean'].append(mean(tir_dict['death hypo']))
            tir_mean_dict['death hypo st dev'].append(stdev(tir_dict['death hypo']))
            tir_mean_dict['ultra hypo mean'].append(mean(tir_dict['ultra hypo']))
            tir_mean_dict['ultra hypo st dev'].append(stdev(tir_dict['ultra hypo']))
            tir_mean_dict['heavy hypo mean'].append(mean(tir_dict['heavy hypo']))
            tir_mean_dict['heavy hypo st dev'].append(stdev(tir_dict['heavy hypo']))
            tir_mean_dict['severe hypo mean'].append(mean(tir_dict['severe hypo']))
            tir_mean_dict['severe hypo st dev'].append(stdev(tir_dict['severe hypo']))
            tir_mean_dict['hypo mean'].append(mean(tir_dict['hypo']))
            tir_mean_dict['hypo st dev'].append(stdev(tir_dict['hypo']))
            tir_mean_dict['time in range mean'].append(mean(tir_dict['time in range']))
            tir_mean_dict['time in range st dev'].append(stdev(tir_dict['time in range']))
            tir_mean_dict['hyper mean'].append(mean(tir_dict['hyper']))
            tir_mean_dict['hyper st dev'].append(stdev(tir_dict['hyper']))
            tir_mean_dict['severe hyper mean'].append(mean(tir_dict['severe hyper']))
            tir_mean_dict['severe hyper st dev'].append(stdev(tir_dict['severe hyper']))
            tir_mean_dict['heavy hyper mean'].append(mean(tir_dict['heavy hyper']))
            tir_mean_dict['heavy hyper st dev'].append(stdev(tir_dict['heavy hyper']))
            tir_mean_dict['ultra hyper mean'].append(mean(tir_dict['ultra hyper']))
            tir_mean_dict['ultra hyper st dev'].append(stdev(tir_dict['ultra hyper']))
            tir_mean_dict['death hyper mean'].append(mean(tir_dict['death hyper']))
            tir_mean_dict['death hyper st dev'].append(stdev(tir_dict['death hyper']))
            tir_mean_dict['LBGI mean of means'].append(mean(tir_dict['LBGI mean']))
            tir_mean_dict['LBGI mean of std'].append(mean_std(tir_dict['LBGI std']))
            tir_mean_dict['HBGI mean of means'].append(mean(tir_dict['HBGI mean']))
            tir_mean_dict['HBGI mean of std'].append(mean_std(tir_dict['HBGI std']))
            tir_mean_dict['RI mean of means'].append(mean(tir_dict['RI mean']))
            tir_mean_dict['RI mean of std'].append(mean_std(tir_dict['RI std']))
            # tir_mean_dict['cap iper mean'].append(tir_dict['cap iper'][0])
            # tir_mean_dict['cap ipo mean'].append(tir_dict['cap ipo'][0])
            # tir_mean_dict['soglia iper mean'].append(tir_dict['soglia iper'][0])
            # tir_mean_dict['soglia ipo mean'].append(tir_dict['soglia ipo'][0])
            # tir_mean_dict['test timesteps'].append(tir_dict['test timesteps'][0])
            tir_mean_dict['ripetizioni'].append(num_games)
            # tir_mean_dict['training learning rate'].append(training_learning_rate)
            # tir_mean_dict['training n steps'].append(training_n_steps)
            # tir_mean_dict['training timesteps'].append(training_total_timesteps)
            # tir_mean_dict['scenario'].append(scenario_usato)
            # tir_mean_dict['start time'].append(start_time)
        
        
            df_cap_mean = pd.DataFrame(tir_mean_dict)

            df_cap_mean.to_excel(final_writer, sheet_name='risultati_finali', index=False)
        
            # final_writer.save()