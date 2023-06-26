# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:34:39 2022

@author: Daniele
"""


import gym
from gym.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from simglucose.simulation.scenario import CustomScenario
from stable_baselines3.ppo.policies import MlpPolicy
# from stable_baselines3.common.evaluation import evaluate_policy
# from gym.wrappers.order_enforcing import OrderEnforcing
# from stable_baselines3.common.monitor import Monitor
import numpy as np
import pandas as pd
import time
import os


from datetime import datetime
date_time = str(datetime.now())[:19].replace(" ", "_" ).replace("-", "" ).replace(":", "" )

model_path = 'C:\GitHub\simglucose\Simulazioni_RL'



def sum_rows_of_xlsx_files(folder_path, string_to_match, df_final):
    
    df_fin = pd.DataFrame()
    
    # crea una lista di tutti i file .xlsx nella cartella
    files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx') and string_to_match in file]

    # inizializza la somma delle righe
    total_rows = 0

    # per ogni file nella lista
    for file in files:
        # carica il file Excel in un DataFrame
        df = pd.read_excel(os.path.join(folder_path, file))
        df_fin = df_fin.append(df)
        os.remove(os.path.join(folder_path, file))
        # calcola il numero di righe del DataFrame (esclusa la prima riga con i label)
        num_rows = len(df.index) -1
        print(file)
        print(num_rows)
        # aggiungi il numero di righe alla somma totale
        total_rows += num_rows
    
    df_fin.to_excel(df_final)
    
    return df_final

def sliding_window(list, window_size):
    
    if len(list) <= window_size:
       return list
    for i in range(len(list)-window_size+1):
        print(np.mean(list[i:i+window_size]))

def quad_func(a,x):
    return -a*(x-90)*(x-150)

def quad_reward(BG_last_hour):
    return quad_func(0.0417, BG_last_hour[-1])

def new_func(x):
    return -0.0417 * x**2 + 10.4167 * x - 525.0017

def new_reward(BG_last_hour):
    return new_func(BG_last_hour[-1])

# exp. function
def exp_func(x,a=0.0417,k=0.3,hypo_treshold = 80, hyper_threshold = 180, exp_bool=True):
  if exp_bool:
    return -a*(x-hypo_treshold)*(x-hyper_threshold) - np.exp(-k*(x-hypo_treshold))
  else:
    return -a*(x-hypo_treshold)*(x-hyper_threshold)

def exp_reward(BG_last_hour,a=0.0417,k=0.3,hypo_treshold = 90, hyper_threshold = 150, exp_bool=True):
    return exp_func(BG_last_hour[-1])

# def create_scenario(n_days, cho_daily=230):
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

# prova cartpole
# env = gym.make('CartPole-v1')
# env = OrderEnforcing(env)

# paziente = 'adolescent#007'

# now = datetime.now() # gestire una qualsiasi data di input
# start_time = datetime.combine(now.date(), datetime.min.time())
start_time = datetime.strptime('3/4/2022 12:00 AM', '%m/%d/%Y %I:%M %p')
# newdatetime = now.replace(hour=12, minute=00)

data = str(datetime.now()).replace(" ", "_" ).replace("-", "" ).replace(":", "" )[:8]

os.chdir('C:\GitHub\simglucose\Simulazioni_RL\Risultati')
cwd = os.getcwd()

data_path = os.path.join(cwd, data)  
if not os.path.exists(data_path):
    os.makedirs(data_path)
    
strategy_path = os.path.join(cwd, 'Strategy')
if not os.path.exists(strategy_path):
    os.makedirs(strategy_path)

model_path = 'C:\GitHub\simglucose\Simulazioni_RL'

logdir = os.path.join(model_path, 'logdir')
if not os.path.exists(logdir):
    os.makedirs(logdir)
    
# tensorboard --logdir=C:\GitHub\simglucose\Simulazioni_RL\logdir --bind_all
    
# logdir = os.path.join(logdir_path, 'logdir')


# dizionario = {'paziente':['adult#001', 'adult#002', 'adult#003' , 'adult#004', 'adult#005',
#                 'adult#006', 'adult#007', 'adult#008' , 'adult#009', 'adult#010'],
#               'ins_max':[0.08, 0.08, 0.08, 0.06, 0.08,
#                         0.09, 0.08, 0.07, 0.07, 0.07]}

# df_cap = pd.DataFrame(dizionario)
# df_cap.to_excel(os.path.join(strategy_path,'paz_cap.xlsx'),index=False)

# dizionario['adult#007'] = 0.11
# ins_max_list = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0,12, 0.12, 0.13, 0.14]
# ins_max_list = [0.12]

# pazienti_list = ['adult#001', 'adult#002', 'adult#003', 'adult#004', 'adult#005', 
#                  'adult#006', 'adult#007', 'adult#008', 'adult#009', 'adult#010']

# tmstp_list = [1440]#, 2048] [1440, 2,400]
# n_steps_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] # 1 non si può
# n_steps_list = [64]
# tmstps_list = [2, 4, 8, 16, 32, 64]
# CAMBIARE DA 5 A 1 SE SI USA 64
# n_steps_list = [480]
# tmstps_list = [480]

tmstps_list = [32768] # 60 giorni
n_steps_list = [2048] # default
# n_steps_list = [32]


opt_dict = {
            'adult#001':['adult#008', 0.07],
            'adult#002':['adult#008', 0.07],
            'adult#003':['adult#007', 0.07],
            'adult#004':['adult#008', 0.07],
            'adult#005':['adult#001', 0.08],
            'adult#006':['adult#003', 0.08],
            'adult#007':['adult#003', 0.08],
            'adult#008':['adult#001', 0.08],
            'adult#009':['adult#003', 0.08],
            'adult#010':['adult#004', 0.07],
            }

for total_timesteps, n_steps in zip(tmstps_list,  n_steps_list):

# for total_timesteps in tmstps_list:
#     for n_steps in n_steps_list:
        for p, cap in list(opt_dict.items()):
        
       
                print('training', p, cap[1])
                
                dizionario = {'paziente': p,
                              'ins_max': cap[1],
                              }
        
                df_cap = pd.DataFrame(dizionario, index=[0])
                df_cap['timesteps'] = total_timesteps
                df_cap['elapsed_time'] = 0
                df_cap.to_excel(os.path.join(strategy_path,'paz_cap.xlsx'),index=False)
                
                paziente = p
                n_days = 60
                # n_hours = n_days*24
                # scen_long = [(12, 100), (20, 120), (23, 30), (31, 40), (36, 70), (40, 100), (47, 10)] # scenario di due giorni
                # if c < 0.09:
                #     ppo_type = 'hypo'
                #     cho_daily=230
                # else:
                #     ppo_type = 'hyper'
                #     cho_daily=190
                # scen_long = create_scenario(n_days, cho_daily=cho_daily)
                scen_long = create_scenario(n_days)
                scenario = CustomScenario(start_time=start_time, scenario=scen_long)#, seed=seed)
                # scenario = CustomScenario(scenario=scen_long)#, seed=seed)

                # registrazione per train singolo
                register(
                    # id='simglucose-adolescent2-v0',
                    id='simglucose-adult2-v0',
                    # entry_point='simglucose.envs:T1DSimEnv',
                    entry_point='simglucose.envs:PPOSimEnv',
                    kwargs={'patient_name': paziente,
                            'reward_fun': new_reward,
                            'custom_scenario': scenario
                            })
    
    
    
                # make env
                env = gym.make('simglucose-adult2-v0')
                env.action_space
                env.reset()
        
                # iperparametri
                gamma = 0.99 #  gamma = 0 -> ritorno nell'immediato futuro
                # 1, 0.99, 0.95, 0.9, 0.7, 0.5
                # gae_gamma # tradeoff bias varianza 0 = maggiore varianza e minor bias (più precisi ma più instabili)
                # net_arch = dict(pi=[64, 64, 64], vf=[64, 64, 64])
                learning_rate = 0.0003
                # learning_rate = 0.00003 # new lr
                model = PPO(MlpPolicy, env, verbose=0, n_steps=n_steps,
                            # batch_size=
                            gamma=gamma,                           
                            learning_rate=learning_rate,
                            tensorboard_log=logdir)
                
                model.load(os.path.join(model_path, 'ppo_withcaps_'+cap[0]+'_nsteps_1024_total_tmstp_1024_lr_00003_insmax'+str(cap[1]).replace('.', '')))
    
                # train
                
                checkpoint_callback = CheckpointCallback(
                      save_freq=2048,
                      save_path=model_path,
                      name_prefix="single_ppo_online_callback_"+p+'_nsteps_'+str(n_steps)+'_total_tmstp_'+str(total_timesteps)+"_lr_"+str(learning_rate).replace('.','')+'_insmax'+str(cap[1]).replace('.',''),
                      save_replay_buffer=True,
                      save_vecnormalize=True,
                    )
                
                model.learn(total_timesteps=total_timesteps, 
                                    callback=checkpoint_callback, progress_bar=True,
                                tb_log_name='PPO', reset_num_timesteps=False)
                
                # env.show_history()
                # start_time = time.perf_counter()
                # for i in range(5):
                #     model.learn(total_timesteps=total_timesteps, progress_bar=True,
                #                 tb_log_name='PPO', reset_num_timesteps=False)
                #     end_time = time.perf_counter()
                #     execution_time = end_time - start_time
    
                #     model.save(os.path.join(model_path, "ppo_online_"+p+'_nsteps_'+str(n_steps)+'_total_tmstp_'+str(total_timesteps)+"_lr_"+str(learning_rate).replace('.','')+'_insmax'+str(c).replace('.','')+'_'+str(total_timesteps*(i+1)))) # single train
                    # model.save(os.path.join(model_path, "ppo_sim_mod_food_hour_"+p+'_tmstp'+str(total_timesteps)+"_lr"+str(learning_rate).replace('.','')+'_insmax'+str(c).replace('.','')+'_'+ppo_type+'_'+str(cho_daily)+'scen'))
    
                # Close the environment
                env.close()
                
                sum_rows_of_xlsx_files(model_path, p+'_'+str(cap[1])+'_history.xlsx', 
                                        os.path.join(model_path, 'continual_learning_'+p+'_trained_on_'+cap[0]+'_'+str(cap[1]).replace('.',''))+'_final_history.xlsx')

# import os
# import openpyxl

# folder_path = "C:/GitHub/simglucose/Simulazioni_RL"  # sostituire con il percorso della cartella da controllare
# total_length = 0


# for filename in os.listdir(folder_path):
#     if filename.endswith("#009_0.05_history.xlsx"):
#         file_path = os.path.join(folder_path, filename)
#         df = pd.read_excel(file_path)
#         file_length = len(df)
#         total_length += file_length

# print("La lunghezza totale dei file è:", total_length)

# import os
# import pandas as pd



# print(sum_rows_of_xlsx_files(model_path, p+'_'+c+'_history.xlsx'))


# import matplotlib.pyplot as plt

# numbers = [79, 94, 79, 74, 103, 78, 88, 69, 62, 58, 57, 56, 62, 59, 58, 72, 68, 64, 67, 74, 87, 78, 76, 68, 73, 87, 97, 61]

# # Crea una lista di valori x da 0 a n-1, dove n è il numero di elementi nella lista numbers
# x_values = range(len(numbers))

# # Crea un grafico a barre
# plt.bar(x_values, numbers)


