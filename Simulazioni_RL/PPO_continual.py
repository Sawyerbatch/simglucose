# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:34:39 2022

@author: Daniele
"""


import gym
from gym.envs.registration import register
from stable_baselines3 import PPO
# from stable_baselines import PPO2
from simglucose.simulation.scenario import CustomScenario
from stable_baselines3.ppo.policies import MlpPolicy, CnnPolicy, MultiInputPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from gym.wrappers.order_enforcing import OrderEnforcing
from stable_baselines3.common.monitor import Monitor
import numpy as np
import pandas as pd
import time
import os

from datetime import datetime
date_time = str(datetime.now())[:19].replace(" ", "_" ).replace("-", "" ).replace(":", "" )


# scenari cumulativi
# dimostrare che approccio continuo non funziona
# addestramento unico con cap iper ottimizzato per ogni paziente con 1440 timesteps
# addestramento unico con cap ipo ottimizzato per ogni paziente con 1440 timesteps

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

now = datetime.now() # gestire una qualsiasi data di input
start_time = datetime.combine(now.date(), datetime.min.time())
newdatetime = now.replace(hour=12, minute=00)

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


dizionario = {'paziente':['adult#001', 'adult#002', 'adult#003' , 'adult#004', 'adult#005',
                'adult#006', 'adult#007', 'adult#008' , 'adult#009', 'adult#010'],
              'ins_max':[0.08, 0.08, 0.08, 0.06, 0.08,
                        0.09, 0.08, 0.07, 0.07, 0.07]}

dizionario = {'paziente':'adult#001',
              'ins_max':[0.08]}

df_cap = pd.DataFrame(dizionario)
df_cap.to_excel(os.path.join(strategy_path,'paz_cap.xlsx'),index=False)



tmstp_list = [1440]#, 2048] [1440, 2,400]


        
paziente = dizionario['paziente']

n_days = 5
n_hours = n_days*24
scen_long = [(12, 100), (20, 120), (23, 30), (31, 40), (36, 70), (40, 100), (47, 10)] # scenario di due giorni
scen_long = create_scenario(n_days)
scenario = CustomScenario(start_time=start_time, scenario=scen_long)#, seed=seed)



# registrazione per train singolo
register(
    # id='simglucose-adolescent2-v0',
    id='simglucose-adult2-v0',
    # entry_point='simglucose.envs:T1DSimEnv',
    entry_point='simglucose.envs:PPOSimEnv',
    kwargs={'patient_name': paziente,
            'reward_fun': new_reward,
            'custom_scenario': scenario})


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

# if paziente == 'adult#001':
#     model = PPO(MlpPolicy, env, verbose=0, gamma=gamma, learning_rate=learning_rate)
# else:
model = PPO.load(os.path.join(model_path, "ppo_sim_mod_food_hour_adult#002_tmstp1440_lr00003_insmax008_customscen"),
                 env=env)


# train
total_timesteps= tmstp_list[0]
# total_timesteps= 1000
start_time = time.perf_counter()
# Train the agent for 10000 steps
model.learn(total_timesteps=total_timesteps, progress_bar=True)
            # reset_num_timesteps=False, )
end_time = time.perf_counter()
execution_time = end_time - start_time


            # model.save(os.path.join(model_path, "ppo_sim_mod_food_hour_"+p+'_tmstp'+str(total_timesteps)+"_lr"+str(learning_rate).replace('.','')+'_insmax'+str(c).replace('.','')+'_customscen')) # single train

model.save(os.path.join(model_path, "ppo_sim_mod_food_hour_continual"))
# Close the environment
env.close()

