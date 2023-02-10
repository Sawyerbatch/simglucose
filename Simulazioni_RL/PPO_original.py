# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:34:39 2022

@author: Daniele
"""

import gym
import numpy as np
from stable_baselines3 import PPO
# from stable_baselines import PPO2
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from gym.wrappers.order_enforcing import OrderEnforcing
from stable_baselines3.common.monitor import Monitor

from datetime import datetime
date_time = str(datetime.now())[:19].replace(" ", "_" ).replace("-", "" ).replace(":", "" )


def quad_func(a,x):
    return -a*(x-70)*(x-180)

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

def exp_reward(BG_last_hour,a=0.0417,k=0.3,hypo_treshold = 80, hyper_threshold = 180, exp_bool=True):
    return exp_func(BG_last_hour[-1])

# paziente = 'adolescent#007'
paziente = 'adult#001'
# env = gym.make('CartPole-v1')
# env = OrderEnforcing(env)
from gym.envs.registration import register
register(
    # id='simglucose-adolescent2-v0',
    id='simglucose-adult2-v0',
    # entry_point='simglucose.envs:T1DSimEnv',
    entry_point='simglucose.envs:PPOSimEnv',
    kwargs={'patient_name': paziente,
            'reward_fun': new_reward}
)


# env = gym.make('simglucose-adolescent2-v0')
env = gym.make('simglucose-adult2-v0')
# env.action_space
env.observation_space
env.reset()

# env = Monitor(env, allow_early_resets=True, override_existing=True)
model = PPO(MlpPolicy, env, verbose=0)

# def evaluate(model, num_episodes=10):
#     """
#     Evaluate a RL agent
#     :param model: (BaseRLModel object) the RL Agent
#     :param num_episodes: (int) number of episodes to evaluate it
#     :return: (float) Mean reward for the last num_episodes
#     """
#     # This function will only work for a single Environment
#     env = model.get_env()
#     all_episode_rewards = []
#     for i in range(num_episodes):
#         episode_rewards = []
#         done = False
#         obs = env.reset()
#         while not done:
#             # _states are only useful when using LSTM policies
#             action, _states = model.predict(obs)
#             # here, action, rewards and dones are arrays
#             # because we are using vectorized env
#             obs, reward, done, info = env.step(action)
#             episode_rewards.append(reward)

#         all_episode_rewards.append(sum(episode_rewards))

#     mean_episode_reward = np.mean(all_episode_rewards)
#     print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

#     return mean_episode_reward
# n_eval_episodes=10
# old_mean_reward, old_std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)#, return_episode_rewards=True)

# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
total_timesteps= 10000
# total_timesteps= 1000

import time
start_time = time.perf_counter()
# Train the agent for 10000 steps
model.learn(total_timesteps=total_timesteps, progress_bar=True)
end_time = time.perf_counter()
execution_time = end_time - start_time


# Evaluate the trained agent
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)

# print(f"old_mean_reward: {old_mean_reward:.2f} +/- {old_std_reward:.2f}")
# print('Step di addestramento: '+str(n_eval_episodes))
# print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
# print('Numero di episodi: '+str(n_eval_episodes))
# print(f'Learning time {execution_time:.6f} seconds in '+str(total_timesteps)+' timesteps')

model.save("ppo_sim_mod_food_hour_"+str(total_timesteps)+"tmstp_"+date_time)

# Close the environment
env.close()

# Set up fake display; otherwise rendering will fail
# import os
# os.system("Xvfb :1 -screen 0 1024x768x24 &")
# os.environ['DISPLAY'] = ':1'

# import base64
# from pathlib import Path

# from IPython import display as ipythondisplay

# def show_videos(video_path='', prefix=''):
#   """
#   Taken from https://github.com/eleurent/highway-env

#   :param video_path: (str) Path to the folder containing videos
#   :param prefix: (str) Filter the video, showing only the only starting with this prefix
#   """
#   html = []
#   for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
#       video_b64 = base64.b64encode(mp4.read_bytes())
#       html.append('''<video alt="{}" autoplay 
#                     loop controls style="height: 400px;">
#                     <source src="data:video/mp4;base64,{}" type="video/mp4" />
#                 </video>'''.format(mp4, video_b64.decode('ascii')))
#   ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))
  
# from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

# def record_video(env_id, model, video_length=500, prefix='', video_folder='videos/'):
#   """
#   :param env_id: (str)
#   :param model: (RL model)
#   :param video_length: (int)
#   :param prefix: (str)
#   :param video_folder: (str)
#   """
#   eval_env = DummyVecEnv([lambda: gym.make(env_id)])
#   # Start the video at step=0 and record 500 steps
#   eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
#                               record_video_trigger=lambda step: step == 0, video_length=video_length,
#                               name_prefix=prefix)

#   obs = eval_env.reset()
#   for _ in range(video_length):
#     action, _ = model.predict(obs)
#     obs, _, _, _ = eval_env.step(action)

#   # Close the video recorder
#   eval_env.close()
  
# record_video('CartPole-v1', model, video_length=500, prefix='ppo-cartpole')

# show_videos('videos', prefix='ppo')

# model = PPO('MlpPolicy', "CartPole-v1", verbose=1).learn(1000)