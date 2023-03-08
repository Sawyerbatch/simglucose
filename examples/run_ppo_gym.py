# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:34:39 2022

@author: Daniele
"""


import gym
from gym.envs.registration import register
from stable_baselines3 import PPO
# from stable_baselines import PPO2
from simglucose.controller.ppo_ctrller import PPOController
from simglucose.simulation.scenario import CustomScenario
from stable_baselines3.ppo.policies import MlpPolicy, CnnPolicy, MultiInputPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from gym.wrappers.order_enforcing import OrderEnforcing
from stable_baselines3.common.monitor import Monitor
import numpy as np
import pandas as pd
import time
import os
from statistics import mean

from datetime import datetime
# date_time = str(datetime.now())[:19].replace(" ", "_" ).replace("-", "" ).replace(":", "" )


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
def create_scenario(n_days, cho_daily=280):

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

# prova cartpole
# env = gym.make('CartPole-v1')
# env = OrderEnforcing(env)

# now = datetime.now() # gestire una qualsiasi data di input
# start_time = datetime.combine(now.date(), datetime.min.time())
# newdatetime = now.replace(hour=12, minute=00)

data = str(datetime.now()).replace(" ", "_" ).replace("-", "" ).replace(":", "" )[:8]

n_days = 5
n_hours = n_days*24
seed = 42
ma = 15
# ripetizioni = 100
# patient_names = ['adult#007']

cgm_name = 'Dexcom'
insulin_pump_name = 'Nuovo'
# start_time = newdatetime
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

# paziente = 'adult#007'
# start_time = random_date(d1, d2)
# n_days = 5
# n_hours = n_days*24
# scen_long = [(12, 100), (20, 120), (23, 30), (31, 40), (36, 70), (40, 100), (47, 10)] # scenario di due giorni
# scen_long = create_scenario(n_days)
# scenario = CustomScenario(start_time=start_time, scenario=scen_long)#, seed=seed)

# pazienti_list = ['adult#001', 'adult#002', 'adult#003', 'adult#004', 'adult#005', 'adult#007', 'adult#008', 'adult#010']

# grid search per ogni paziente, ogni combinazione deve avere 10 ripetizioni e dare la media
# su 006, 009 e 010

# cap iper da 0.09 a 0.13
# cap ipo da 0.05 e 0.08
# soglia iper 160 e 165
# soglia ipo 80,85,90


tir_dict = {'time in range mean':[],
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
            'start':[]}


# opt_dict = {'adult#001':('009','006',160,85,100),
#             # 'adult#002':('014','008',165,85),
#             # 'adult#003':('011','006',160,90),
#             # 'adult#004':('009','005',165,95),
#             # 'adult#005':('013','008',165,90),
#             # 'adult#006':('015','007',170,95),
#             'adult#006':('015','007',170,95,100),
#             # 'adult#007':('011','008',160,85),
#             # 'adult#007':('011','005',160,85),
#             # 'adult#007':('011','006',160,85),
#             # 'adult#007':('011','007',160,85),
#             # 'adult#007':('011','007',160,80),
#             # 'adult#008':('01','006',160,95),
#             'adult#009':('014','005',170,85, 100),
#             'adult#009':('014','005',170,85,100),
#             # 'adult#010':('014','007',160,90)
#             }
pazienti_list = ['adult#009']#, 'adult#010', 'adult#006'] #['adult#006','adult#009','adult#010']
iper_control_list = ['009', '01', '011', '012', '013']
ipo_control_list = ['005','006', '007', '008']
iper_soglia_list = [160,165]
ipo_soglia_list = [80,85,90]
ripetizioni = 10

# per provare
# iper_control_list = ['009']
# ipo_control_list = ['005','006']
# iper_soglia_list = [160]
# ipo_soglia_list = [80]
# ripetizioni = 10
for paziente in pazienti_list:
    
    df_final = pd.DataFrame(columns=list(tir_dict.keys()))

    for iper in iper_control_list:
        # print('iper cap: '+iper)
        for ipo in ipo_control_list:
            # print('ipo cap: '+ipo)
            for iper_s in iper_soglia_list:
                # print('iper soglia: '+str(iper_s))
                for ipo_s in ipo_soglia_list:
                # print('ipo soglia: '+str(ipo_s))
                # opt_dict = {'adult#006':(iper,ipo,iper_s,ipo_s, ripetizioni),
                #             'adult#009':(iper,ipo,iper_s,ipo_s, ripetizioni),
                #             'adult#010':(iper,ipo,iper_s,ipo_s, ripetizioni),
                #             }


# for paziente in pazienti_list:#['adult#006', 'adult#009']:#, 'adult#008']:
    
                # for k,v in opt_dict.items():
                    
                #     paziente = k
                #     iper = v[0]
                #     ipo = v[1]
                #     iper_s = v[2]
                #     ipo_s = v[3]
                #     ripetizioni = v[4]
                    
                    print(paziente, iper, ipo, iper_s, ipo_s)
                    
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
                                'start':[]}
                    
                    for i in range(ripetizioni):
                
                        start_time = random_date(d1, d2)
                        n_days = 5
                        n_hours = n_days*24
                        scen_long = [(12, 100), (20, 120), (23, 30), (31, 40), (36, 70), (40, 100), (47, 10)] # scenario di due giorni
                        scen_long = create_scenario(n_days)
                        scenario = CustomScenario(start_time=start_time, scenario=scen_long)#, seed=seed)
                
                        
                        # iper_control_list = ['012', '013', '014']
                        # ipo_control_list = ['005','006', '007', '008']
                        # iper_soglia_list = [160,165,170]
                        # ipo_soglia_list = [85,90,95]
                        timesteps = 2400 # 5 giorni
                        # timesteps = 480 # 1 giorno
                        # timesteps = 10000 
                        training = 1440
                        # iper_control_list = ['01']
                        # ipo_control_list = ['006']
                        # iper_soglia_list = [160]
                        # ipo_soglia_list = [95]
                        
                        
                        # registrazione per train singolo
                        register(
                            # id='simglucose-adolescent2-v0',
                            id='simglucose-adult2-v0',
                            # entry_point='simglucose.envs:T1DSimEnv',
                            entry_point='simglucose.envs:PPOSimEnv',
                            kwargs={'patient_name': paziente,
                                    'reward_fun': new_reward,
                                    'custom_scenario': scenario})
                        
                        # for iper in iper_control_list:
                        #     print('iper cap: '+iper)
                        #     for ipo in ipo_control_list:
                        #         print('ipo cap: '+ipo)
                        #         for iper_s in iper_soglia_list:
                        #             print('iper soglia: '+str(iper_s))
                        #             for ipo_s in ipo_soglia_list:
                        #                 print('ipo soglia: '+str(ipo_s))
                        
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
                        
                        # best result 007: 006, 008, 90, 165
                        
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
                            
                            # if tir[0] > 3.0 or tir[1] > 9.0 or tir[3] > 40.0 or tir[4] > 15.0:
                            # # if done:
                            #     print("Episode finished after {} timesteps".format(t + 1))
                            #     break
                        
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
                        tir_dict['start'].append(start_time)
                        
                        # df_cap = pd.DataFrame(tir_dict)
                        # df_cap.to_excel(os.path.join(strategy_path,'performance_'+paziente+'_'+iper+'_'+ipo+'_'+str(iper_s)+'_'+str(ipo_s)+'_'+str(timesteps)+'_train1440(2048)_ripetizioni'+str(ripetizioni)+'.xlsx')
                                        # ,sheet_name='results', index=False)
                    
                    tir_mean_dict = dict()
                    tir_mean_dict['time in range mean'] = mean(tir_dict['time in range'])
                    tir_mean_dict['hyper mean'] = mean(tir_dict['hyper'])
                    tir_mean_dict['hypo mean'] = mean(tir_dict['hypo'])
                    tir_mean_dict['severe hyper mean'] = mean(tir_dict['severe hyper'])
                    tir_mean_dict['severe hypo mean'] = mean(tir_dict['severe hypo'])
                    tir_mean_dict['cap iper mean'] = tir_dict['cap iper'][0]
                    tir_mean_dict['cap ipo mean'] = tir_dict['cap ipo'][0]
                    tir_mean_dict['soglia iper mean'] = tir_dict['soglia iper'][0]
                    tir_mean_dict['soglia ipo mean'] = tir_dict['soglia ipo'][0]
                    tir_mean_dict['timesteps'] = tir_dict['timesteps'][0]
                    tir_mean_dict['ripetizioni'] = tir_dict['ripetizioni'][0]
                    tir_mean_dict['start'] = tir_dict['start'][0]

                    df_cap_mean = pd.DataFrame(tir_mean_dict, index=([0]))
                    # df_cap_mean.to_excel(os.path.join(strategy_path,'performance_'+paziente+'_'+str(timesteps)+'_train1440(2048)_100test.xlsx')
                    #                 ,sheet_name='results mean', index=False)
                    
                    df_final = df_final.append(df_cap_mean, ignore_index = True)
                    
                    df_final.to_excel(os.path.join(strategy_path,'performance_'+paziente+'_'+str(timesteps)+'_allmeans_train1440(2048)_100test.xlsx')
                                    ,sheet_name='risultati', index=False)
                
                    # with pd.ExcelWriter(os.path.join(strategy_path,'performance_'+paziente+'_'+iper+'_'+ipo+'_'+str(iper_s)+'_'+str(ipo_s)+'_'+str(timesteps)+'_train1440(2048)_ripetizioni'+str(ripetizioni)+'.xlsx')) as writer:
                       
                        # use to_excel function and specify the sheet_name and index
                        # to store the dataframe in specified sheet
                        # df_cap.to_excel(writer, sheet_name="results", index=False)
                        # df_cap_mean.to_excel(writer, sheet_name="results mean", index=False)
                        # data_frame3.to_excel(writer, sheet_name="Baked Items", index=False)    
    