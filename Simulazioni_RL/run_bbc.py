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
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController



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


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def insert_dot(string, index):
    return string[:index] + '.' + string[index:]


data = str(datetime.now()).replace(" ", "_" ).replace("-", "" ).replace(":", "" )[:8]


# test parameters

# n_days = 10
# n_hours = n_days*24
test_timesteps = 2400 # 5 giorni
# test_timesteps = 4800 # 10 giorni
start_time = datetime.strptime('3/4/2022 12:00 AM', '%m/%d/%Y %I:%M %p')
seed = 42
ma = 1
ripetizioni = 100



os.chdir('C:\GitHub\simglucose\Simulazioni_RL\Risultati')
cwd = os.getcwd()
    
strategy_path = os.path.join(cwd, 'Strategy')
if not os.path.exists(strategy_path):
    os.makedirs(strategy_path)

model_path = 'C:\GitHub\simglucose\Simulazioni_RL'

scenario_usato = '30_days_1000_times'

with open(os.path.join(model_path, 'Risultati\Strategy', 'scenarios_'+scenario_usato+'.json')) as json_file:
    scenarios = json.load(json_file)


pazienti = ['adult#001', 'adult#002', 'adult#003', 'adult#004',
            'adult#005', 'adult#006', 'adult#007', 'adult#008',
            'adult#009', 'adult#010']

# pazienti = ['adult#007']


writer = pd.ExcelWriter(os.path.join(strategy_path,'performance_bbc_test_timesteps_'+str(test_timesteps)+'_ripetizioni_'+str(ripetizioni)+'.xlsx'))

tir_mean_dict = {
            'paziente':[],
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
            'test timesteps':[],
            'ripetizioni':[],
            'scenario':[],
            'start time': []
            }


for paziente in pazienti:
    
    strategy = 'BBC'
    df_strategy = pd.DataFrame({'strategy': strategy, 'patient': [paziente]})

    df_strategy.to_excel(os.path.join(strategy_path,'strategy.xlsx'),index=False)
                 
    print(paziente)
    
    tir_dict = {'time in range':[],
                'hyper':[],
                'hypo':[],
                'severe hyper':[],
                'severe hypo':[],
                'test timesteps':[],
                'ripetizione':[],
                'scenario':[],
                }
   
    for i, scen in zip(range(ripetizioni), scenarios.values()):
        
        
        scen = [tuple(x) for x in scen]
        scenario = CustomScenario(start_time=start_time, scenario=scen)
           
        # simglucose parameters
        patient = T1DPatient.withName(paziente)
        sensor = CGMSensor.withName('Dexcom', seed=1)
        pump = InsulinPump.withName('Insulet')
        # cgm_name = 'Dexcom'
        # insulin_pump_name = 'Insulet'
        animate = True
        parallel = True
        controller = BBController()
        # env = T1DSimEnv(paziente, cgm_name, insulin_pump_name, scenario, strategy)
        
        env = T1DSimEnv(patient=patient,
                        sensor=sensor,
                        pump = pump,
                        scenario=scenario)
        
        observation, reward, done, info = env.reset()
        
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
            
            action = controller.policy(observation, reward, done, **info) # BBC
 
            observation, reward, done, info = env.step(action)
  
            if observation.CGM < 50:
                counter_50 += 1
            if 50 <= observation.CGM < 70:
                counter_70 += 1
            if 70 <= observation.CGM <= 180:
                counter_180 += 1
            if 180 < observation.CGM <= 250:
                counter_250 += 1
            if observation.CGM > 250:
                counter_over_250 += 1
            
            counter_total += 1
            print(paziente)
            print(i+1)
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
        tir_dict['test timesteps'].append(test_timesteps)
        tir_dict['ripetizione'].append(i+1)
        tir_dict['scenario'].append(scen)

        df_cap = pd.DataFrame(tir_dict)
        
        df_cap.to_excel(writer, sheet_name=paziente, index=False)


    tir_mean_dict['paziente'].append(paziente)
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
    tir_mean_dict['test timesteps'].append(tir_dict['test timesteps'][0])
    tir_mean_dict['ripetizioni'].append(ripetizioni)
    tir_mean_dict['scenario'].append(scenario_usato)
    tir_mean_dict['start time'].append(start_time)


    df_cap_mean = pd.DataFrame(tir_mean_dict)
    
    df_cap_mean.to_excel(writer, sheet_name='risultati', index=False)
    
    writer.save()