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


# dizionario = {'paziente':['adult#001', 'adult#002', 'adult#003' , 'adult#004', 'adult#005',
#                 'adult#006', 'adult#007', 'adult#008' , 'adult#009', 'adult#010'],
#               'ins_max':[0.08, 0.08, 0.08, 0.06, 0.08,
#                         0.09, 0.08, 0.07, 0.07, 0.07]}

# df_cap = pd.DataFrame(dizionario)
# df_cap.to_excel(os.path.join(strategy_path,'paz_cap.xlsx'),index=False)

# dizionario['adult#007'] = 0.11
# ins_max_list = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0,12, 0.12, 0.13, 0.14]
# ins_max_list = [0.12]

# pazienti_list = ['adult#001', 'adult#002', 'adult#003', 'adult#004', 'adult#005', 
#                  'adult#006', 'adult#007', 'adult#008', 'adult#009', 'adult#010']

tmstp_list = [1440]#, 2048] [1440, 2,400]

opt_dict = {
            # 'adult#001':[0.07],
            # 'adult#002':[0.07],
            # 'adult#003':[0.07],
            # 'adult#004':[0.07],
            # 'adult#005':[0.07],
            # 'adult#006':[0.14],
            # 'adult#007':[0.7],
            # 'adult#008':[0.07],
            'adult#009':[0.04,0.03],
            # 'adult#010':[0.05,0.06,0.08,0.09,0.10,0.11,0.12,0.13,0.14],
            }

# pazienti_list = ['adult#006', 'adult#009']#, 'adult#002', 'adult#008']#', 'adult#010']

# for p in pazienti_list:
#     for i in ins_max_list:
    
for t in tmstp_list:
    for p, cap in list(opt_dict.items()):
    
        for c in cap:
   
            print('training', p, c)
            
            dizionario = {'paziente': p,
                          'ins_max': c}
    
            df_cap = pd.DataFrame(dizionario, index=[0])
            df_cap.to_excel(os.path.join(strategy_path,'paz_cap.xlsx'),index=False)
        
    
            # paziente = 'adult#007'
            paziente = p
            n_days = 5
            n_hours = n_days*24
            # scen_long = [(12, 100), (20, 120), (23, 30), (31, 40), (36, 70), (40, 100), (47, 10)] # scenario di due giorni
            scen_long = create_scenario(n_days)
            scenario = CustomScenario(start_time=start_time, scenario=scen_long)#, seed=seed)

# df_cap = pd.DataFrame(dizionario)
# df_cap.to_excel(os.path.join(strategy_path,'paz_cap.xlsx'),index=False)
# cap = df_cap.loc[df_cap['paziente']==p].iloc[:,1]
# cap = cap.iloc[0]
# cap = float(df_cap.loc[df_cap['paziente']=='adult#010'].iloc[:,1])

            # registrazione per train singolo
            register(
                # id='simglucose-adolescent2-v0',
                id='simglucose-adult2-v0',
                # entry_point='simglucose.envs:T1DSimEnv',
                entry_point='simglucose.envs:PPOSimEnv',
                kwargs={'patient_name': paziente,
                        'reward_fun': new_reward,
                        'custom_scenario': scenario})



# train con tutti
# pazienti = ['adult#00'+str(i) for i in list(range(1,10))]+['adult#010']
# for paziente in pazienti:
#     register(
#         # id='simglucose-adolescent2-v0',
#         id='simglucose-adult2-v0',
#         # entry_point='simglucose.envs:T1DSimEnv',
#         entry_point='simglucose.envs:PPOSimEnv',
#         kwargs={'patient_name': paziente,
#                 'reward_fun': new_reward})
    
#     env = gym.make('simglucose-adult2-v0')
#     # env.reset()
        
#     if paziente == 'adult#001':
    
#         model = PPO(MlpPolicy, env, verbose=0)
#     else:
#         # model = PPO.load("ppo_sim_mod_food_hour_ALL_10000tmstp")
#         path = 'C:\GitHub\simglucose\Simulazioni_RL'
#         # loading the saved model
#         model = PPO.load(os.path.join(path, "ppo_sim_mod_food_hour_ALL_10000tmstp"))
    
#     total_timesteps= 10000
#     model.learn(total_timesteps=total_timesteps, progress_bar=True)
#     model.save("ppo_sim_mod_food_hour_ALL_10000tmstp")
        
        
# env = Monitor(env, allow_early_resets=True, override_existing=True)

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
            model = PPO(MlpPolicy, env, verbose=0, gamma=gamma, learning_rate=learning_rate)

# model = PPO(MlpPolicy, env, verbose=0)
# model = PPO(CnnPolicy, env, verbose=0)
# model = PPO(MultiInputPolicy, env, verbose=0)

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

            # train
            total_timesteps= t
            # total_timesteps= 1000
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

            model.save(os.path.join(model_path, "ppo_sim_mod_food_hour_"+p+'_tmstp'+str(total_timesteps)+"_lr"+str(learning_rate).replace('.','')+'_insmax'+str(c).replace('.','')+'_customscen')) # single train


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