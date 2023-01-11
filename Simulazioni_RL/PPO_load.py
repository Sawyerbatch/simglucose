# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:34:39 2022

@author: Daniele
"""

import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from gym.wrappers.order_enforcing import OrderEnforcing
from stable_baselines3.common.monitor import Monitor

def quad_func(a,x):
    return -a*(x-70)*(x-180)

def quad_reward(BG_last_hour):
    return quad_func(0.01, BG_last_hour[-1])

# def custom_reward(BG_last_hour):
#     if BG_last_hour[-1] > 180:
#         return -1
#     elif BG_last_hour[-1] < 70:
#         return -2
#     else:
#         return 1

# env = gym.make('CartPole-v1')
# env = OrderEnforcing(env)
paziente = 'adolescent#007'
from gym.envs.registration import register
register(
    id='simglucose-adolescent2-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': paziente,
            'reward_fun': quad_reward}
)

env = gym.make('simglucose-adolescent2-v0')
# env.action_space
env.observation_space
env.reset()

# env = Monitor(env, allow_early_resets=True, override_existing=True)
# model = PPO(MlpPolicy, env, verbose=0)

# loading the saved model
model = PPO.load("ppo_sim_mod")

# # get the first observation of the environment
# obs = env.reset()

# predict an action
# action, _states = model.predict(obs)

# close the environment
# env.close()
n_eval_episodes=10
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
print('Paziente: '+ paziente)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
print('Numero di episodi: '+str(n_eval_episodes))
# print(f'Learning time {execution_time:.6f} seconds in '+str(total_timesteps)+' timesteps')