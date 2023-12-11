# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:07:29 2023

@author: Daniele
"""

import pettingzoo
import tianshou
from tianshou.env import PettingZooEnv
# from tianshou.env import MultiAgentEnv
import gymnasium
from gymnasium.envs.registration import register

register(
    id="simglucose/adolescent2-v0",
    entry_point="simglucose.envs:T1DSimGymnasiumEnv_MARL",
    max_episode_steps=10,
    kwargs={"patient_name": "adolescent#002"},
)

env = PettingZooEnv(gymnasium.make("simglucose/adolescent2-v0", render_mode="human"))
# env = MultiAgentEnv(gymnasium.make("simglucose/adolescent2-v0", render_mode="human"))
# env = gymnasium.make("simglucose/adolescent2-v0", render_mode="human")


observation, info = env.reset()
for t in range(200):
    env.render()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(
        f"Step {t}: observation {observation}, reward {reward}, terminated {terminated}, truncated {truncated}, info {info}"
    )
    if terminated or truncated:
        print("Episode finished after {} timesteps".format(t + 1))
        break