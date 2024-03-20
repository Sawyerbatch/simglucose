from __future__ import annotations

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:09:12 2024

@author: Daniele
"""

"""Uses Stable-Baselines3 to train agents to play the Waterworld environment using SuperSuit vector envs.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""


import glob
import os
import time
from typing import Optional
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from pettingzoo.sisl import waterworld_v4

import gymnasium
import time
import warnings
import json
from datetime import datetime
from stable_baselines3 import PPO
from simglucose.envs import T1DSimGymnasiumEnv_MARL
from simglucose.simulation.scenario import CustomScenario

# Disable all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def new_func(x):
    return -0.0417 * x**2 + 10.4167 * x - 525.0017

def new_reward(BG_last_hour):
    return new_func(BG_last_hour[-1])

start_time = datetime.strptime('3/4/2022 12:00 AM', '%m/%d/%Y %I:%M %p')

with open('scenarios_5_days_1000_times.json') as json_file:
    scenarios = json.load(json_file)
    
scen = list(scenarios.values())[0]
scen = [tuple(x) for x in scen]
scenario = CustomScenario(start_time=start_time, scenario=scen)


def train_butterfly_supersuit(
    env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs
):
    # Train a single model to play as each agent in a cooperative Parallel environment
    # env = env_fn.parallel_env(**env_kwargs)
    env = env_fn

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")

    # Note: Waterworld's observation space is discrete (242,) so we use an MLP policy rather than CNN
    model = PPO(
        MlpPolicy,
        env,
        verbose=3,
        learning_rate=1e-3,
        batch_size=256,
    )

    model.learn(total_timesteps=steps, progress_bar=True)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


# def eval(env_fn, num_games: int = 100, render_mode: Optional[str] = None, **env_kwargs):
#     # Evaluate a trained agent vs a random agent
#     env = env_fn
#     # env = env_fn.env(render_mode=render_mode, **env_kwargs)

#     print(
#         f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
#     )

#     try:
#         latest_policy = max(
#             glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
#         )
#     except ValueError:
#         print("Policy not found.")
#         exit(0)

#     model = PPO.load(latest_policy)

#     rewards = {agent: 0 for agent in env.possible_agents}

#     # Note: We train using the Parallel API but evaluate using the AEC API
#     # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
#     for i in range(num_games):
#         env.reset(seed=i)

#         for agent in env.agent_iter():
#             obs, reward, termination, truncation, info = env.last()
#             print(agent, obs, reward)
#             for a in env.agents:
#                 rewards[a] += env.rewards[a]
#             if termination or truncation:
#                 break
#             else:
#                 act = model.predict(obs, deterministic=True)[0]

#             env.step(act)
#     env.close()

#     avg_reward = sum(rewards.values()) / len(rewards.values())
#     print("Rewards: ", rewards)
#     print(f"Avg reward: {avg_reward}")
#     return avg_reward

def eval(env_fn, num_games: int = 100, render_mode: Optional[str] = None, **env_kwargs):
    # Initialize the environment
    env = env_fn  # Assicurati che env_fn restituisca un'istanza dell'ambiente

    print(f"\nStarting evaluation on {env.metadata['name']} (num_games={num_games}, render_mode={render_mode})")

    try:
        latest_policy = max(glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime)
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    total_rewards = {agent: 0 for agent in env.possible_agents}
    
    for i in range(num_games):
        obs = env.reset()  # Resetta l'ambiente e ottieni l'osservazione iniziale
        observ = obs[0]
        done = False
        while not done:
            actions = {}
 
            # print(a, type(a))
            for agent, agent_obs in observ.items():
                action, _ = model.predict(agent_obs, deterministic=True)  # Ottieni l'azione per ogni agente
                actions[agent] = action
            
            obs, rewards, dones, truncs, infos = env.step(actions)  # Esegui un passo dell'ambiente con le azioni degli agenti
            
            for agent, reward in rewards.items():
                total_rewards[agent] += reward  # Aggiorna il totale dei premi
            
            done = all(dones.values())  # Controlla se tutti gli agenti hanno terminato
    
    env.close()

    avg_reward = sum(total_rewards.values()) / len(total_rewards.values())
    print("Total rewards:", total_rewards)
    print(f"Average reward: {avg_reward}")
    return avg_reward

if __name__ == "__main__":
    
    env_fn = T1DSimGymnasiumEnv_MARL(
        patient_name='adult#001',
        custom_scenario=scenario,
        reward_fun=new_reward,
        seed=123,
        render_mode="human",
    )
    
    # env_fn = waterworld_v4
    env_kwargs = {}

    # Train a model (takes ~3 minutes on GPU)
    train_butterfly_supersuit(env_fn, steps=3, seed=0, **env_kwargs)

    # Evaluate 10 games (average reward should be positive but can vary significantly)
    eval(env_fn, num_games=1, render_mode=None, **env_kwargs)

    # Watch 2 games
    eval(env_fn, num_games=1, render_mode="human", **env_kwargs)