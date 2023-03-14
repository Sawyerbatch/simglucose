# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:34:39 2022

@author: Daniele
"""


import gym
from gym.envs.registration import register
from stable_baselines3 import PPO
from simglucose.simulation.scenario import CustomScenario
import numpy as np
import pandas as pd
import os
from statistics import mean
from datetime import datetime
from random import randrange
from datetime import timedelta

def random_date(start, end):
    """
    This function will return a random datetime between two datetime 
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)

d1 = datetime.strptime('1/1/2022 1:30 PM', '%m/%d/%Y %I:%M %p')
d2 = datetime.strptime('1/1/2023 4:50 AM', '%m/%d/%Y %I:%M %p')


# date_time = str(a)[:19].replace(" ", "_" ).replace("-", "" ).replace(":", "" )



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
  # cho_sum = 0
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

def insert_dot(string, index):
    return string[:index] + '.' + string[index:]


data = str(datetime.now()).replace(" ", "_" ).replace("-", "" ).replace(":", "" )[:8]

n_days = 5
n_hours = n_days*24
seed = 42
ma = 15


cgm_name = 'Dexcom'
insulin_pump_name = 'Nuovo'
animate = True
parallel = True


os.chdir('C:\GitHub\simglucose\Simulazioni_RL\Risultati')
cwd = os.getcwd()

data_path = os.path.join(cwd, data)  
if not os.path.exists(data_path):
    os.makedirs(data_path)
    
strategy_path = os.path.join(cwd, 'Strategy')
if not os.path.exists(strategy_path):
    os.makedirs(strategy_path)

model_path = 'C:\GitHub\simglucose\Simulazioni_RL'


pazienti_list = ['adult#009']#, 'adult#010', 'adult#006'] #['adult#006','adult#009','adult#010']
# iper_control_list = ['009', '01', '011', '012', '013']
# ipo_control_list = ['005','006', '007', '008']
# iper_soglia_list = [160,165]
# ipo_soglia_list = [80,85,90]
# ripetizioni = 10

# per provare
iper_control_list = ['013']
ipo_control_list = ['004']
iper_soglia_list = [170]
ipo_soglia_list = [75]
ripetizioni = 100

# iper_control_list = ['013', '014', '015', '016', '017', '018','019', '02']
# ipo_control_list = ['003','004','005']
# iper_soglia_list = [130, 140, 150 ,160,170]
# ipo_soglia_list = [60, 70, 75, 80, 90,]
# ripetizioni = 50


labels = [
            'time in range mean',
            'hyper mean',
            'hypo mean',
            'severe hyper mean',
            'severe hypo mean',
            'cap iper mean',
            'cap ipo mean',
            'soglia iper mean',
            'soglia ipo mean',
            'timesteps',
            'ripetizioni',
]



df_final = pd.DataFrame(columns=labels)


for paziente in pazienti_list:
    
    for iper in iper_control_list:
        print('iper cap: '+iper)
        for ipo in ipo_control_list:
            print('ipo cap: '+ipo)
            for iper_s in iper_soglia_list:
                print('iper soglia: '+str(iper_s))
                for ipo_s in ipo_soglia_list:
                    print('ipo soglia: '+str(ipo_s))
                    
                    tir_mean_dict = {

                                'time in range mean':[],
                                'hyper mean':[],
                                'hypo mean':[],
                                'severe hyper mean':[],
                                'severe hypo mean':[],
                                'cap iper mean':[],
                                'cap ipo mean':[],
                                'soglia iper mean':[],
                                'soglia ipo mean':[],
                                'timesteps':[],
                                'ripetizioni':[],
                                }
                    
                   
                    for i in range(ripetizioni):
                        
                        tir_dict = {'time in range':[],
                                    'hyper':[],
                                    'hypo':[],
                                    'severe hyper':[],
                                    'severe hypo':[],
                                    'cap iper':[],
                                    'cap ipo':[],
                                    'soglia iper':[],
                                    'soglia ipo':[],
                                    'timesteps':[],
                                    'ripetizioni':[],
                
                                    }
                
                        start_time = random_date(d1, d2)
                        n_days = 5
                        n_hours = n_days*24
                        scen_long = create_scenario(n_days)
                        scenario = CustomScenario(start_time=start_time, scenario=scen_long)#, seed=seed)
                
                        timesteps = 2400 # 5 giorni
                
                        training = 1440
                
                        # registrazione per train singolo
                        register(
                            # id='simglucose-adolescent2-v0',
                            id='simglucose-adult2-v0',
                            # entry_point='simglucose.envs:T1DSimEnv',
                            entry_point='simglucose.envs:PPOSimEnv',
                            kwargs={'patient_name': paziente,
                                    'reward_fun': new_reward,
                                    'custom_scenario': scenario})
                
                        
                        model_ppo_iper = PPO.load(os.path.join(model_path, 'ppo_sim_mod_food_hour_'+paziente+'_tmstp'+str(training)+'_lr00003_insmax'+iper+'_customscen')) # iper  
                        model_ppo_ipo = PPO.load(os.path.join(model_path, 'ppo_sim_mod_food_hour_'+paziente+'_tmstp'+str(training)+'_lr00003_insmax'+ipo+'_customscen'))  # ipo   
                    
                        env = gym.make('simglucose-adult2-v0')
                        
                        observation = env.reset()
                        
                        cgm_list = list()
                        counter_50 = 0
                        counter_70 = 0
                        counter_180 = 0
                        counter_250 = 0
                        counter_over_250 = 0
                        counter_total = 0
                        
                        tir = np.zeros(shape=(5,))
                        
                        # ogni timestep equivale a 3 minuti
                        for t in range(timesteps):
                            
                            # env.render(mode='human')
                            
                            print(observation)
                    
                            if observation[0][0] < ipo_s:
                                action = np.array([[0.0]])
                            elif observation[0][0] > iper_s:
                                action = model_ppo_iper.predict(np.array(observation)) # iper control
                            else:
                                action = model_ppo_ipo.predict(np.array(observation)) # ipo control
                            observation, reward, done, info = env.step(action[0])
                            if observation[0][0] < 50:
                                counter_50 += 1
                            if 50 <= observation[0][0] < 70:
                                counter_70 += 1
                            if 70 <= observation[0][0] <= 180:
                                counter_180 += 1
                            if 180 < observation[0][0] <= 250:
                                counter_250 += 1
                            if observation[0][0] > 250:
                                counter_over_250 += 1
                            
                            counter_total += 1
                            
                            print(paziente)
                            print(i)
                            tir[0] = (counter_50/counter_total)*100
                            print('severe hypo:',tir[0])
                            tir[1] = (counter_70/counter_total)*100
                            print('hypo:', tir[1])
                            tir[2] = (counter_180/counter_total)*100
                            print('time in range:', tir[2])
                            tir[3] = (counter_250/counter_total)*100
                            print('hyper:', tir[3])
                            tir[4] = (counter_over_250/counter_total)*100
                            print('severe hyper:', tir[4])
                            
                        
                        tir_dict['time in range'].append(tir[2])
                        tir_dict['hyper'].append(tir[3])
                        tir_dict['hypo'].append(tir[1])
                        tir_dict['severe hyper'].append(tir[4])
                        tir_dict['severe hypo'].append(tir[0])
                        iper_mod = insert_dot(iper, 1)
                        tir_dict['cap iper'].append(iper_mod)
                        ipo_mod = insert_dot(ipo, 1)
                        tir_dict['cap ipo'].append(ipo_mod)
                        tir_dict['soglia iper'].append(iper_s)
                        tir_dict['soglia ipo'].append(ipo_s)
                        tir_dict['timesteps'].append(timesteps)
                        tir_dict['ripetizioni'].append(ripetizioni)
                
                    
                    tir_mean_dict['time in range mean'].append(mean(tir_dict['time in range']))
                    tir_mean_dict['hyper mean'].append(mean(tir_dict['hyper']))
                    tir_mean_dict['hypo mean'].append(mean(tir_dict['hypo']))
                    tir_mean_dict['severe hyper mean'].append(mean(tir_dict['severe hyper']))
                    tir_mean_dict['severe hypo mean'].append(mean(tir_dict['severe hypo']))
                    tir_mean_dict['cap iper mean'].append(tir_dict['cap iper'][0])
                    tir_mean_dict['cap ipo mean'].append(tir_dict['cap ipo'][0])
                    tir_mean_dict['soglia iper mean'].append(tir_dict['soglia iper'][0])
                    tir_mean_dict['soglia ipo mean'].append(tir_dict['soglia ipo'][0])
                    tir_mean_dict['timesteps'].append(tir_dict['timesteps'][0])
                    tir_mean_dict['ripetizioni'].append(tir_dict['ripetizioni'][0])
                
                    df_cap_mean = pd.DataFrame(tir_mean_dict)
                
                    df_final = df_final.append(df_cap_mean)
                                
                    df_final.to_excel(os.path.join(strategy_path,'performance_gridsearch_'+paziente+'_'+str(timesteps)+'_train1440(2048)_'+str(ripetizioni)+'ripetizioni_singola_combinazione.xlsx')
                                ,sheet_name='risultati', index=False)
