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
from statistics import mean, stdev
from datetime import datetime
from random import randrange
from datetime import timedelta
import json
import scipy

# def random_date(start, end):
#     """
#     This function will return a random datetime between two datetime 
#     objects.
#     """
#     delta = end - start
#     int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
#     random_second = randrange(int_delta)
#     return start + timedelta(seconds=random_second)

# d1 = datetime.strptime('1/1/2022 1:30 PM', '%m/%d/%Y %I:%M %p')
# d2 = datetime.strptime('1/1/2023 4:50 AM', '%m/%d/%Y %I:%M %p')



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

# def create_scenario(n_days, cho_daily=280):
# def create_scenario(n_days, cho_daily=230):

#   scenario = []
#   # cho_sum = 0
#   mu_break, sigma_break = 8, 3 
#   mu_lunch, sigma_lunch = 13, 1
#   mu_snack, sigma_snack = 17, 2
#   mu_dinner, sigma_dinner = 21, 2
#   mu_night, sigma_night = 24, 2

#   for i in range(n_days):

#     mu_cho_break, sigma_cho_break = cho_daily*0.15, 15
#     mu_cho_lunch, sigma_cho_lunch = cho_daily*0.45, 45
#     mu_cho_snack, sigma_cho_snack = cho_daily*0.05, 5
#     mu_cho_dinner, sigma_cho_dinner = cho_daily*0.35, 35
#     mu_cho_night, sigma_cho_night = cho_daily*0.05, 5

#     hour_break = int(np.random.normal(mu_break, sigma_break/2)) + 24*i
#     hour_lunch = int(np.random.normal(mu_lunch, sigma_lunch/2)) + 24*i
#     hour_snack = int(np.random.normal(mu_snack, sigma_snack/2)) + 24*i
#     hour_dinner = int(np.random.normal(mu_dinner, sigma_dinner/2)) + 24*i
#     hour_night = int(np.random.normal(mu_night, sigma_night/2)) + 24*i

#     cho_break = int(np.random.normal(mu_cho_break, sigma_cho_break/2))
#     cho_lunch = int(np.random.normal(mu_cho_lunch, sigma_cho_lunch/2))
#     cho_snack = int(np.random.normal(mu_cho_snack, sigma_cho_snack/2))
#     cho_dinner = int(np.random.normal(mu_cho_dinner, sigma_cho_dinner/2))
#     cho_night = int(np.random.normal(mu_cho_night, sigma_cho_night/2))

#     if int(np.random.randint(100)) < 60:
#       scenario.append((hour_break,cho_break))
#     if int(np.random.randint(100)) < 100:
#       scenario.append((hour_lunch,cho_lunch))
#     if int(np.random.randint(100)) < 30:
#       scenario.append((hour_snack,cho_snack))
#     if int(np.random.randint(100)) < 95:
#       scenario.append((hour_dinner,cho_dinner))
#     if int(np.random.randint(100)) < 3:
#       scenario.append((hour_night,cho_night))

#     #cho_sum += cho_break + cho_lunch + cho_snack + cho_dinner + cho_night

#   return scenario

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def insert_dot(string, index):
    return string[:index] + '.' + string[index:]

os.chdir('C:\GitHub\simglucose\Simulazioni_RL\Risultati')
cwd = os.getcwd()
    
strategy_path = os.path.join(cwd, 'Strategy')
if not os.path.exists(strategy_path):
    os.makedirs(strategy_path)

model_path = 'C:\GitHub\simglucose\Simulazioni_RL'

scenario_usato = '5_days_1000_times'

with open(os.path.join(model_path, 'Risultati\Strategy', 'scenarios_'+scenario_usato+'.json')) as json_file:
    scenarios = json.load(json_file)


data = str(datetime.now()).replace(" ", "_" ).replace("-", "" ).replace(":", "" )[:8]

# training parameters

training_learning_rate = '00003'
training_n_steps = 64
training_total_timesteps = 64
learning_days = [64]#,2880,3360,3840,4320,4800]


# test parameters

n_days = 5
n_hours = n_days*24
test_timesteps = 2400 # 5 giorni
start_time = datetime.strptime('3/4/2022 12:00 AM', '%m/%d/%Y %I:%M %p')
seed = 42
ma = 15
ripetizioni = 50


cgm_name = 'Dexcom'
insulin_pump_name = 'Nuovo'
animate = True
parallel = True


opt_dict = {
            'adult#001':('009','006',160,85),
            'adult#002':('014','008',165,85),
            'adult#003':('011','006',160,90),
            'adult#004':('009','005',165,95),
            'adult#005':('013','008',165,90),
            'adult#006':('015','007',170,95),
            'adult#007':('011','008',160,80),
            'adult#008':('01','006',160,95),
            'adult#009':('013','004',160,60),
            'adult#010':('014','007',160,90)
            }

for k,v in opt_dict.items():

    paziente = k
    iper = v[0]
    ipo = v[1]
    iper_s = v[2]
    ipo_s = v[3]
    
    
    writer = pd.ExcelWriter(os.path.join(strategy_path,'performance_continual_learning_patient_'+k+'_test_timesteps_'+str(test_timesteps)+'_training_nsteps_'+str(training_n_steps)+'_training_tmstp_'+str(training_total_timesteps)+'_ripetizioni_'+str(ripetizioni)+'.xlsx'))
    # df_final = pd.read_excel(os.path.join(strategy_path,'performance_continual_learning_patient_'+k+'_test_timesteps_'+str(test_timesteps)+'_training_nsteps_'+str(training_n_steps)+'_training_tmstp_'+str(training_total_timesteps)+'_ripetizioni_'+str(ripetizioni)+'.xlsx'), engine='openpyxl')
                
    print(paziente, iper, ipo, iper_s, ipo_s)

    tir_mean_dict = {
                'learning_days':[],
                'time in range mean':[],
                'time in range st dev':[],
                'hyper mean':[],
                'hyper st dev':[],
                'hypo mean':[],
                'hypo st dev':[],
                'severe hyper mean':[],
                'severe hyper st dev':[],
                'severe hypo mean':[],
                'severe hypo st dev':[],
                'cap iper mean':[],
                'cap ipo mean':[],
                'soglia iper mean':[],
                'soglia ipo mean':[],
                'training learning rate':[],
                'training n steps':[],
                'training timesteps':[],
                'test timesteps':[],
                'ripetizioni':[],
                'scenario':[],
                'start time': []
                }

    
    for l in learning_days:
        
        tir_dict = {'time in range':[],
                    'hyper':[],
                    'hypo':[],
                    'severe hyper':[],
                    'severe hypo':[],
                    'cap iper':[],
                    'cap ipo':[],
                    'soglia iper':[],
                    'soglia ipo':[],
                    'training learning rate':[],
                    'training n steps':[],
                    'training timesteps':[],
                    'test timesteps':[],
                    'ripetizione':[],
                    'scenario':[],
                    }
    
        for i, scen in zip(range(ripetizioni), scenarios.values()):
            
            
  

            # start_time = random_date(d1, d2)
            # scen_long = create_scenario(n_days)
            # scenario = CustomScenario(start_time=start_time, scenario=scen_long)#, seed=seed)
            scen = [tuple(x) for x in scen]
            scenario = CustomScenario(start_time=start_time, scenario=scen)
            
            # registrazione per train singolo
            register(
                # id='simglucose-adolescent2-v0',
                id='simglucose-adult2-v0',
                # entry_point='simglucose.envs:T1DSimEnv',
                entry_point='simglucose.envs:PPOSimEnv',
                kwargs={'patient_name': paziente,
                        'reward_fun': new_reward,
                        'custom_scenario': scenario})
    
            
            model_ppo_iper = PPO.load(os.path.join(model_path, "ppo_online_"+paziente+'_nsteps_'+str(training_n_steps)+'_total_tmstp_'+str(training_total_timesteps)+"_lr_"+training_learning_rate+'_insmax'+iper+'_'+str(l))) # iper  
            model_ppo_ipo = PPO.load(os.path.join(model_path, "ppo_online_"+paziente+'_nsteps_'+str(training_n_steps)+'_total_tmstp_'+str(training_total_timesteps)+"_lr_"+training_learning_rate+'_insmax'+ipo+'_'+str(l)))  # ipo   
        
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
            for t in range(test_timesteps):
                
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
                print(i+1)
                print(l)
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
            tir_dict['test timesteps'].append(test_timesteps)
            tir_dict['ripetizione'].append(i+1)
            tir_dict['training learning rate'].append(training_learning_rate)
            tir_dict['training n steps'].append(training_n_steps)
            tir_dict['training timesteps'].append(training_total_timesteps)
            tir_dict['scenario'].append(scen)
            
            df_cap = pd.DataFrame(tir_dict)       
            df_cap.to_excel(writer, sheet_name='learning_days_'+str(l), index=False)
    
        
        tir_mean_dict['learning_days'].append(l)
        tir_mean_dict['time in range mean'].append(mean(tir_dict['time in range']))
        tir_mean_dict['time in range st dev'].append(stdev(tir_dict['time in range']))
        tir_mean_dict['hyper mean'].append(mean(tir_dict['hyper']))
        tir_mean_dict['hyper st dev'].append(stdev(tir_dict['hyper']))
        tir_mean_dict['hypo mean'].append(mean(tir_dict['hypo']))
        tir_mean_dict['hypo st dev'].append(stdev(tir_dict['hypo']))
        tir_mean_dict['severe hyper mean'].append(mean(tir_dict['severe hyper']))
        tir_mean_dict['severe hyper st dev'].append(stdev(tir_dict['severe hyper']))
        tir_mean_dict['severe hypo mean'].append(mean(tir_dict['severe hypo']))
        tir_mean_dict['severe hypo st dev'].append(stdev(tir_dict['severe hypo']))
        tir_mean_dict['cap iper mean'].append(tir_dict['cap iper'][0])
        tir_mean_dict['cap ipo mean'].append(tir_dict['cap ipo'][0])
        tir_mean_dict['soglia iper mean'].append(tir_dict['soglia iper'][0])
        tir_mean_dict['soglia ipo mean'].append(tir_dict['soglia ipo'][0])
        tir_mean_dict['test timesteps'].append(tir_dict['test timesteps'][0])
        tir_mean_dict['ripetizioni'].append(ripetizioni)
        tir_mean_dict['training learning rate'].append(training_learning_rate)
        tir_mean_dict['training n steps'].append(training_n_steps)
        tir_mean_dict['training timesteps'].append(training_total_timesteps)
        tir_mean_dict['scenario'].append(scenario_usato)
        tir_mean_dict['start time'].append(start_time)
    
    
        df_cap_mean = pd.DataFrame(tir_mean_dict)
        
        df_cap_mean.to_excel(writer, sheet_name='risultati', index=False)
        
        writer.save()
