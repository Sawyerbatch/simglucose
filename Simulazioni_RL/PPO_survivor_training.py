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



from stable_baselines3.common.callbacks import BaseCallback


class CustomCallback(BaseCallback):
    def __init__(self, p, target_timesteps=4800, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.p = p
        self.target_timesteps = target_timesteps
        
        
    def _on_step(self) -> bool:
        # Ottieni la durata dell'episodio corrente
        # episode_length = self.locals.get("episode").length
        episode_length = self.num_timesteps
        
        # Controlla se la durata dell'episodio raggiunge la soglia desiderata
        if episode_length >= self.target_timesteps:
            self.model.save("ppo_survivor_callback_"+self.p+"_target_"+str(self.target_timesteps))
            return False

        return True

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
    

# tmstps_list = [32768] # 60 giorni
total_timesteps = 100000
target_timesteps = 4800 #960
# n_steps_list = [2048] # default
# n_steps_list = [32]


opt_dict = {
            # 'adult#001':0.08,
            # 'adult#002':0.08,
            # 'adult#003':0.08,
            # 'adult#004':0.05,
            # 'adult#005':0.08,
            # 'adult#006':0.08,
            # 'adult#007':0.06,
            # 'adult#008':0.07,
            # 'adult#009':0.08,
            # 'adult#010':0.08
            }

# for total_timesteps, n_steps in zip(tmstps_list,  n_steps_list):

# for total_timesteps in tmstps_list:
#     for n_steps in n_steps_list:
for p, cap in list(opt_dict.items()):
    
   
    print('training', p, cap)
    
    dizionario = {'paziente': p,
                  'ins_max': cap,
                  }

    df_cap = pd.DataFrame(dizionario, index=[0])
    df_cap['timesteps'] = total_timesteps
    df_cap['target timesteps'] = target_timesteps
    df_cap['elapsed_time'] = 0
    df_cap.to_excel(os.path.join(strategy_path,'paz_cap.xlsx'),index=False)
    
    paziente = p
    n_days = 10
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
    model = PPO(MlpPolicy, env, verbose=0, # n_steps default 2048,
                # batch_size=
                gamma=gamma,                           
                learning_rate=learning_rate,
                tensorboard_log=logdir)
    
    # model.load(os.path.join(model_path, 'ppo_withcaps_'+cap[0]+'_nsteps_1024_total_tmstp_1024_lr_00003_insmax'+str(cap[1]).replace('.', '')))

    # train
    
    # checkpoint_callback = CheckpointCallback(
    #       save_freq=2048,
    #       save_path=model_path,
    #       name_prefix="ppo_survivor_callback_"+p+'_nsteps_'+str(n_steps)+'_total_tmstp_'+str(total_timesteps)+"_lr_"+str(learning_rate).replace('.','')+'_insmax'+str(cap[1]).replace('.',''),
    #       save_replay_buffer=True,
    #       save_vecnormalize=True,
    #     )
    
    # model.learn(total_timesteps=total_timesteps, 
    #                     callback=checkpoint_callback, progress_bar=True,
    #                 tb_log_name='PPO', reset_num_timesteps=False)
    
    callback = CustomCallback(p, target_timesteps)
    
    model.learn(total_timesteps=total_timesteps, 
                        callback=callback, progress_bar=True,
                    tb_log_name='PPO', reset_num_timesteps=False)
    
    # Close the environment
    env.close()
    
    sum_rows_of_xlsx_files(model_path, p+'_'+str(cap)+'_history.xlsx', 
                            os.path.join(model_path, 'survivor_continual_learning_'+str(target_timesteps)+'_'+p+'_'+str(cap).replace('.',''))+'_final_history.xlsx')



