# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:07:29 2023

@author: Daniele
"""

# import pettingzoo
# import tianshou
# from tianshou.env import PettingZooEnv
# from tianshou.env import MultiAgentEnv
# import gymnasium
# from gymnasium.envs.registration import register
# from pettingzoo import ParallelEnv
from simglucose.envs import T1DSimGymnasiumEnv_MARL
# from pettingzoo.test import parallel_api_test, api_test
import warnings
import json
from simglucose.simulation.scenario import CustomScenario
# from tianshou.env import PettingZooEnv
# import pettingzoo
# from pettingzoo import AECEnv
from datetime import datetime
# import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy

# # Definisci una classe wrapper per adattare l'ambiente multi-agente a pettingzoo.AECEnv
# class T1DSimGymnasiumWrapper(AECEnv):
#     def __init__(self, original_env):
#         self.env = original_env

#     def reset(self):
#         return self.env.reset()

#     def observe(self, agent):
#         # Implementa la logica per ottenere le osservazioni specifiche per l'agente
#         pass

#     def step(self, action):
#         return self.env.step(action)

#     def render(self, mode='human'):
#         return self.env.render(mode)

# Disable all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# register(
#     id="simglucose/adolescent2-v0",
#     entry_point="simglucose.envs:T1DSimGymnasiumEnv_MARL",
#     max_episode_steps=10,
#     kwargs={"patient_name": "adolescent#002"},
# )

# # Crea l'ambiente multi-agente e il wrapper
# original_env = T1DSimGymnasiumEnv_MARL()
# wrapped_env = T1DSimGymnasiumWrapper(original_env)
# # Utilizza l'ambiente avvolto
# env = PettingZooEnv(wrapped_env)


# env = PettingZooEnv(T1DSimGymnasiumEnv_MARL())
# env = T1DSimGymnasiumEnv_MARL()
# env = PettingZooEnv("simglucose/adolescent2-v0")
# env = PettingZooEnv(gymnasium.make("simglucose/adolescent2-v0", render_mode="human"))
# env = MultiAgentEnv(gymnasium.make("simglucose/adolescent2-v0", render_mode="human"))
# env = gymnasium.make("simglucose/adolescent2-v0", render_mode="human")


# observation, info = env.reset()
# print(observation, info)


# api_test(env)



# for t in range(200):
#     env.render()
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)
#     print(
#         f"Step {t}: observation {observation}, reward {reward}, terminated {terminated}, truncated {truncated}, info {info}"
#     )
#     if terminated or truncated:
#         print("Episode finished after {} timesteps".format(t + 1))
#         break

#%%

import time

def new_func(x):
    return -0.0417 * x**2 + 10.4167 * x - 525.0017

def new_reward(BG_last_hour):
    return new_func(BG_last_hour[-1])

start_time = datetime.strptime('3/4/2022 12:00 AM', '%m/%d/%Y %I:%M %p')

with open('scenarios_5_days_1000_times.json') as json_file:
    scenarios = json.load(json_file)
    
scen = list(scenarios.values())[0]
    
scen = [tuple(x) for x in scen]
# scen[0]  = (1,30)
scenario = CustomScenario(start_time=start_time, scenario=scen)

# register(
#         # id='simglucose-adolescent2-v0',
#         id='simglucose-adult2-v0',
#         # entry_point='simglucose.envs:T1DSimEnv',
#         entry_point='simglucose.envs:T1DSimGymnasiumEnv_MARL',
#         kwargs={'patient_name': 'adult#001',\
#                 'reward_fun': new_reward,
#                 'custom_scenario': scenario
#                 })

    
# Create a list of environment names
# env_names = ['simglucose-adult2-v0'] * 2  # Adjust the number of environments as needed

# Create a parallel environment
# env = ParallelEnv(env_names)
    
# env = PettingZooEnv("simglucose-adult2-v0")
# env = ParallelEnv("simglucose-adult2-v0")
# env = ParallelEnv()
# env = gym.make('simglucose-adult2-v0')
# env = T1DSimGymnasiumEnv_MARL()

# env = pettingzoo.make('simglucose-adult2-v0')

# import gym

# class CustomEnvWrapper(gym.Env):
#     def __init__(self, original_env):
#         self.env = original_env

#     def step(self, action):
#         return self.env.step(action)

#     def reset(self):
#         return self.env.reset()

#     def render(self, mode='human'):
#         return self.env.render(mode)

# # Now, wrap your environment
# env = CustomEnvWrapper(T1DSimGymnasiumEnv_MARL(patient_name='adult#001',
# custom_scenario=scenario,
# # custom_scenario=None,
# reward_fun=new_reward,
# seed=123,
# render_mode="human"))

env = T1DSimGymnasiumEnv_MARL(
    patient_name='adult#001',
    custom_scenario=scenario,
    # custom_scenario=None,
    reward_fun=new_reward,
    seed=123,
    render_mode="human",
)

# # Definisci il numero di passi da eseguire nella simulazione
num_steps = 5000


#%%

gamma = 0.99 #  gamma = 0 -> ritorno nell'immediato futuro
# 1, 0.99, 0.95, 0.9, 0.7, 0.5
# gae_gamma # tradeoff bias varianza 0 = maggiore varianza e minor bias (più precisi ma più instabili)
# net_arch = dict(pi=[64, 64, 64], vf=[64, 64, 64])
learning_rate = 0.0003
# learning_rate = 0.00003 # new lr
model = PPO(MlpPolicy, env, verbose=0, n_steps=num_steps,
            gamma=gamma, learning_rate=learning_rate)

total_timesteps = 1000

model.learn(total_timesteps=total_timesteps, progress_bar=True)



#%%

# parallel_api_test(env)

observation, info = env.reset(seed=42)

# Esegui la simulazione
for step in range(num_steps):
    print('Step numero', step)
    # Esegui un passo dell'ambiente
    # actions = {agent: env.action_space[agent].sample() for agent in env.agents}
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    print(actions)
    observations, rewards, done, truncations, infos = env.step(actions)
    # print(observations)
    # Aggiorna l'ambiente e visualizza le informazioni
    env.render()
    time.sleep(0.1)

    # Verifica se la simulazione è terminata
    if any(done.values()):
        print(f"La simulazione è terminata dopo {step+1} passi")
        break