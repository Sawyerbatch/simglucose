# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:35:20 2022

@author: Daniele
"""

import gym

from stable_baselines3 import A2C

from gym.envs.registration import register
register(
    id='simglucose-adolescent2-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adolescent#002'}
)

env = gym.make('simglucose-adolescent2-v0')

# model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10_000)

# obs = env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()