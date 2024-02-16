# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:07:29 2023

@author: Daniele
"""

# import pettingzoo
# import tianshou
from tianshou.env import PettingZooEnv
# from tianshou.env import MultiAgentEnv
import gymnasium
from gymnasium.envs.registration import register
from pettingzoo import ParallelEnv
from simglucose.envs import T1DSimGymnasiumEnv_MARL
from pettingzoo.test import parallel_api_test, api_test
import warnings


from tianshou.env import PettingZooEnv
from pettingzoo import AECEnv

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
env = T1DSimGymnasiumEnv_MARL()
# env = PettingZooEnv("simglucose/adolescent2-v0")
# env = PettingZooEnv(gymnasium.make("simglucose/adolescent2-v0", render_mode="human"))
# env = MultiAgentEnv(gymnasium.make("simglucose/adolescent2-v0", render_mode="human"))
# env = gymnasium.make("simglucose/adolescent2-v0", render_mode="human")


observation, info = env.reset()
print(observation, info)

parallel_api_test(env)
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

# ...

# Definisci il numero di passi da eseguire nella simulazione
num_steps = 10

# Esegui la simulazione
for step in range(num_steps):
    # Esegui un passo dell'ambiente
    # actions = {agent: env.action_space[agent].sample() for agent in env.agents}
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, done, truncations, infos = env.step(actions)

    # Aggiorna l'ambiente e visualizza le informazioni
    env.render()
    time.sleep(0.1)

    # Verifica se la simulazione è terminata
    if any(done.values()):
        print(f"La simulazione è terminata dopo {step+1} passi")
        break