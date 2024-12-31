# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:34:39 2022

@author: Daniele
"""
import os
import gym
import numpy as np
import time
from datetime import datetime
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.env import T1DSimEnv as T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from gym.wrappers.order_enforcing import OrderEnforcing
from stable_baselines3.common.monitor import Monitor
from statistics import mean, stdev

def moving_average(lst, window_size):
    moving_average_list = []
    window_len = min(len(lst), window_size)
    for i in range(window_len):
        moving_average_head = sum(lst[0:i+1]) / (i+1)
        moving_average_list.append(moving_average_head)
    if window_size > len(lst):            
        for i in range(len(lst) - window_size):
            moving_average = sum(lst[i:i + window_size]) / window_size
            moving_average_list.append(moving_average)
    
    return moving_average_list

def quad_func(a,x):
    return -a*(x-70)*(x-180)

def quad_reward(BG_last_hour):
    return quad_func(0.01, BG_last_hour[-1])

def new_func(x):
    return -0.0417 * x**2 + 10.4167 * x - 525.0017

def new_reward(BG_last_hour):
    return new_func(BG_last_hour[-1])

# def custom_reward(BG_last_hour):
#     if BG_last_hour[-1] > 180:
#         return -1
#     elif BG_last_hour[-1] < 70:
#         return -2
#     else:
#         return 1

# env = gym.make('CartPole-v1')
# env = OrderEnforcing(env)
paziente = 'adult#007'
from gym.envs.registration import register
register(
    id='simglucose-adult2-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': paziente,
            'reward_fun': new_reward}
)

now = datetime.now() # gestire una qualsiasi data di input
newdatetime = now.replace(hour=12, minute=00)

env = gym.make('simglucose-adult2-v0')

# from stable_baselines3.common.env_checker import check_env
# check_env(env)

scen = [(7, 45), (12, 70), (16, 15), (18, 80), (23, 10)]
scenario = CustomScenario(start_time=newdatetime, scenario=scen)
patient = T1DPatient.withName('adult#001')
sensor = CGMSensor.withName('Dexcom', seed=1)
pump = InsulinPump.withName('Nuovo')
# scenario = RandomScenario(start_time=start_time, seed=1)
env = T1DSimEnv(patient, sensor, pump, scenario)
# env.action_space
# env.observation_space
env.reset()

# env = Monitor(env, allow_early_resets=True, override_existing=True)
# model = PPO(MlpPolicy, env, verbose=0)
path = 'C:\GitHub\simglucose\Simulazioni_RL'
# loading the saved model
model = PPO.load(os.path.join(path, "ppo_sim_mod_food_hour_10000tmstp_insmax010"))

# # get the first observation of the environment
obs = env.reset()
# obs = obs[0]

azioni = list()
mean_actions = list()
stati = list()
osservazioni = list()
ricompense = list()
# predict an action

n_eval_episodes=100

for i in range(n_eval_episodes):
    # action, _states = model.predict(obs)
    # obs =  np.expand_dims(obs[0], axis=0)
    # action = model.predict(obs)
    # obs, rewards, dones, info = env.step(action[0])   
    
    # azioni.append(action)
    # # stati.append(_states)
    # osservazioni.append(obs)
    # ricompense.append(rewards)
    
    
    # action, _states = model.predict(obs)
    # obs =  np.expand_dims(obs[0], axis=0)
    action = model.predict(obs)
    
    window_size = 30
    azioni.append(action[0])
    mean_action = moving_average(azioni, window_size)
    obs, rewards, dones, info = env.step(mean_action[-1])  
    mean_actions.append(mean_action[-1])
    # stati.append(_states)
    osservazioni.append(obs)
    ricompense.append(rewards)

# fare tanti predict e fare la media delle ricompense è come valutare la policy?

temp_mean_reward, temp_std_reward = mean(ricompense), stdev(ricompense)
# close the environment
# env.close()
#%%
# old_mean_reward, old_std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)#, return_episode_rewards=True)

# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
# total_timesteps= 100
# import time
# start_time = time.perf_counter()
# # Train the agent for 10000 steps
# model.learn(total_timesteps=total_timesteps)
# end_time = time.perf_counter()
# execution_time = end_time - start_time


# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)

# print(f"old_mean_reward:{old_mean_reward:.2f} +/- {old_std_reward:.2f}")
# print('Paziente: '+ paziente)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
# print('Numero di episodi: '+str(n_eval_episodes))
# print(f'Learning time {execution_time:.6f} seconds in '+str(total_timesteps)+' timesteps')

print(f"temp_mean_reward:{temp_mean_reward:.2f} +/- {temp_std_reward:.2f}")

print(f"ppo_mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")