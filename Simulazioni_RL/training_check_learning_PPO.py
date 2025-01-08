# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:34:39 2022

@author: Daniele
"""


import gym
from gym.envs.registration import register
from stable_baselines3 import PPO
import math
from simglucose.simulation.scenario import CustomScenario
from stable_baselines3.ppo.policies import MlpPolicy
# from stable_baselines3.common.evaluation import evaluate_policy
# from gym.wrappers.order_enforcing import OrderEnforcing
# from stable_baselines3.common.monitor import Monitor
import numpy as np
import pandas as pd
import time
import os
import warnings
from datetime import datetime


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

os.chdir('C:\\Users\\utente\\Documents\\GitHub\\simglucose\\Simulazioni_RL')
cwd = os.getcwd()


# PARAMETRI DA SETTARE

# patient_type = 'adult' # 'adult'

# reward_type = 'new' # 'magni'

n_steps_list = [1024]
total_timesteps_list = [10240] # 10240

n_days = 5
n_hours = n_days*24

ripetizioni = 10

for patient_type in ['adult']:   # 'adolescent'
    for reward_type in ['new', 'magni']:
    
        if reward_type == 'new':
            
            model_path = 'C:\\Users\\utente\\Documents\\GitHub\\simglucose\\Simulazioni_RL\\modelli'
            
            print('using new function')
        
            def new_func(x):
                return -0.0417 * x**2 + 10.4167 * x - 525.0017
            
            def new_reward(BG_last_hour):
                return new_func(BG_last_hour[-1])
        
        
        elif reward_type == 'magni':
            
            model_path = 'C:\\Users\\utente\\Documents\\GitHub\\simglucose\\Simulazioni_RL\\modelli_magni'
            
            print('using magni function')
            
            def clip_0_15_5(x):
                return min(15.5, max(0, x))
            
            def magni_risk(b):
                c0 = 1.509
                c1 = 1.084
                c2 = 5.381
                
                if b < 70:
                    return -1.0
                else:
                    #exponent = c1 - c2
                    inner = (c0 * ((math.log(b))**c1 - c2))
                    clipped_value = clip_0_15_5(10 * (inner**2))
                    return 1 - (clipped_value / 7.75)
            
            def new_reward(BG_last_hour):
                b = BG_last_hour[-1]
                return magni_risk(b)
        
        
        
        # def quad_func(a,x):
        #     return -a*(x-90)*(x-150)
        
        # def quad_reward(BG_last_hour):
        #     return quad_func(0.0417, BG_last_hour[-1])
        
        
        # exp. function
        # def exp_func(x,a=0.0417,k=0.3,hypo_treshold = 80, hyper_threshold = 180, exp_bool=True):
        #   if exp_bool:
        #     return -a*(x-hypo_treshold)*(x-hyper_threshold) - np.exp(-k*(x-hypo_treshold))
        #   else:
        #     return -a*(x-hypo_treshold)*(x-hyper_threshold)
        
        # def exp_reward(BG_last_hour,a=0.0417,k=0.3,hypo_treshold = 90, hyper_threshold = 150, exp_bool=True):
        #     return exp_func(BG_last_hour[-1])
        
        
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
        
        
        now = datetime.now() # gestire una qualsiasi data di input
        start_time = datetime.combine(now.date(), datetime.min.time())
        newdatetime = now.replace(hour=12, minute=00)
        
        # data = str(datetime.now()).replace(" ", "_" ).replace("-", "" ).replace(":", "" )[:8]
        
        # data_path = os.path.join(cwd, data)  
        # if not os.path.exists(data_path):
        #     os.makedirs(data_path)
            
        strategy_path = os.path.join(cwd, 'Strategy')
        if not os.path.exists(strategy_path):
            os.makedirs(strategy_path)
        
        
        if patient_type == 'adult' and reward_type == 'new':
            
            opt_dict = {
                patient_type+'#001': [0.09, 0.06, 0.08],
                patient_type+'#002': [0.14, 0.08],
                patient_type+'#003': [0.11, 0.06, 0.08],
                patient_type+'#004': [0.09, 0.05, 0.07],
                patient_type+'#005': [0.13, 0.08],
                patient_type+'#006': [0.15, 0.07, 0.09],
                patient_type+'#007': [0.11, 0.07],
                patient_type+'#008': [0.10, 0.06, 0.07],
                patient_type+'#009': [0.14, 0.06, 0.07],
                patient_type+'#010': [0.14, 0.07]
        }
            
        elif patient_type == 'adult' and reward_type == 'magni':
            
            {
            patient_type+'#001': [0.12, 0.07, 0.08],
            patient_type+'#002': [0.14, 0.06, 0.07],
            patient_type+'#003': [0.09, 0.08],
            patient_type+'#004': [0.10, 0.07, 0.05],
            patient_type+'#005': [0.12, 0.08, 0.09],
            patient_type+'#006': [0.12, 0.08],
            patient_type+'#007': [0.11, 0.06],
            patient_type+'#008': [0.11, 0.06],
            patient_type+'#009': [0.12, 0.05, 0.08],
            patient_type+'#010': [0.09, 0.08]
        }
            
            
        
        elif patient_type == 'aolescent' and reward_type == 'new':
            
            opt_dict = {
                patient_type+'#001': [0.12, 0.06, 0.05],
                patient_type+'#002': [0.11, 0.06],
                patient_type+'#003': [0.12, 0.05],
                patient_type+'#004': [0.11, 0.07, 0.06],
                patient_type+'#005': [0.09, 0.07, 0.05],
                patient_type+'#006': [0.09, 0.06, 0.05],
                patient_type+'#007': [0.09, 0.06, 0.05],
                patient_type+'#008': [0.11, 0.05, 0.07],
                patient_type+'#009': [0.12, 0.06, 0.05],
                patient_type+'#010': [0.09, 0.05, 0.06]
            }
            
        
        if patient_type == 'adolescent' and reward_type == 'magni':
            
            opt_dict = {
                patient_type+'#001': [0.01, 0.06, 0.05],
                patient_type+'#002': [0.09, 0.07, 0.06],
                patient_type+'#003': [0.11, 0.05],
                patient_type+'#004': [0.09, 0.07, 0.05],
                patient_type+'#005': [0.09, 0.06, 0.07],
                patient_type+'#006': [0.10, 0.06, 0.07],
                patient_type+'#007': [0.09, 0.08, 0.06],
                patient_type+'#008': [0.14, 0.07],
                patient_type+'#009': [0.09, 0.05],
                patient_type+'#010': [0.09, 0.06, 0.05]
            }
                    
        for n_steps, total_timesteps in zip(n_steps_list, total_timesteps_list):
            
                for p, cap in list(opt_dict.items()):
                
                    for n, c in enumerate(cap):
                        
                        for i in range(1,ripetizioni+1):
               
                            print('training', p, c)
                            
                            dizionario = {'paziente': p,
                                          'ins_max': c}
                    
                            df_cap = pd.DataFrame(dizionario, index=[0])
                            df_cap['timesteps'] = total_timesteps
                            df_cap['target timesteps'] = total_timesteps
                            df_cap['elapsed_time'] = 0
                            df_cap['pazient type'] = patient_type
                            df_cap['reward type'] = reward_type
                            df_cap['ripetizione'] = i
                            df_cap['ppo conf'] = 'single'
                            df_cap['check_learning'] = 'Yes'
                            
                            # if n < 2:
                            #     df_cap['ppo conf'] = 'learning_check'
                            # else:
                            #     df_cap['ppo conf'] = 'single'
                                
                            df_cap.to_excel(os.path.join(strategy_path,'paz_cap.xlsx'),index=False)
                            
                            paziente = p
            
                            scen_long = create_scenario(n_days)
                            scenario = CustomScenario(start_time=start_time, scenario=scen_long)#, seed=seed)
                
                            # registrazione per train singolo
                            register(
                                # id='simglucose-adolescent2-v0',
                                id='simglucose-'+patient_type+'2-v0',
                                # entry_point='simglucose.envs:T1DSimEnv',
                                entry_point='simglucose.envs:PPOSimEnv',
                                kwargs={'patient_name': paziente,
                                        'reward_fun': new_reward,
                                        'custom_scenario': scenario})
                
                
                            # make env
                            env = gym.make('simglucose-'+patient_type+'2-v0')
                            env.action_space
                    
                            # iperparametri
                            gamma = 0.99 #  gamma = 0 -> ritorno nell'immediato futuro
                            # 1, 0.99, 0.95, 0.9, 0.7, 0.5
                            # gae_gamma # tradeoff bias varianza 0 = maggiore varianza e minor bias (più precisi ma più instabili)
                            # net_arch = dict(pi=[64, 64, 64], vf=[64, 64, 64])
                            learning_rate = 0.0003
                            # learning_rate = 0.00003 # new lr
                            model = PPO(MlpPolicy, env, verbose=0, n_steps=n_steps,
                                        gamma=gamma, learning_rate=learning_rate)
                
                            # train
            
                            model.learn(total_timesteps=total_timesteps, progress_bar=True)
            
                            # Close the environment
                            env.close()

