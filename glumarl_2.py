# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 01:32:20 2024

@author: Daniele
"""

import time
from datetime import datetime
import json
import numpy as np
from tianshou.data import Batch
from tianshou.env import SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from simglucose.simulation.scenario import CustomScenario
from gymnasium.envs.registration import register
from simglucose.envs import T1DSimGymnasiumEnv_MARL

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

register(
    id='simglucose-adult2-v0',
    entry_point='simglucose.envs:T1DSimGymnasiumEnv_MARL',
    kwargs={'patient_name': 'adult#001',
            'reward_fun': new_reward,
            'custom_scenario': scenario
            }
)

# Creare l'ambiente
env = T1DSimGymnasiumEnv_MARL(
    patient_name='adult#007',
    custom_scenario=scenario,
    reward_fun=new_reward,
    seed=123,
    render_mode="human",
)

# Creare l'ambiente vettorizzato
env = SubprocVectorEnv([lambda: env for _ in range(2)])

# Inizializzare il modello di politica (PPOPolicy è utilizzato come esempio)
policy = PPOPolicy(env)

# Definire altri iperparametri e configurazioni di addestramento
# ...

# Numero totale di episodi da eseguire
num_episodes = 2

# Numero di passi per ogni episodio
num_steps_per_episode = 10

for episode in range(num_episodes):
    print(f'Inizio episodio {episode + 1}')

    # Reset dell'ambiente all'inizio di ogni episodio
    observation, info = env.reset()

    # Liste per raccogliere dati durante l'episodio
    obs_list, act_list, rew_list, done_list = [], [], [], []

    # Eseguire il numero desiderato di passi all'interno di ogni episodio
    for step in range(num_steps_per_episode):
        print(f'Step numero {step + 1}')

        # Esegui un passo dell'ambiente
        actions = policy(observation)
        observations, rewards, done, truncations, infos = env.step(actions)

        # Raccogli dati
        obs_list.append(observation)
        act_list.append(actions)
        rew_list.append(rewards)
        done_list.append(done)

        # Verifica se la simulazione è terminata
        if any(done):
            print(f"La simulazione è terminata dopo {step+1} passi nell'episodio {episode + 1}")
            break

    # # Converti i dati raccolti in Batch per l'addestramento
    # batch = Batch(
    #     obs=obs_list,
    #     act=act_list,
    #     rew=rew_list,
    #     done=np.array(done_list),
    #     obs_next=observations
    # )

        # Esegui l'aggiornamento del modello
        result = policy.learn()

        # Stampa informazioni sull'addestramento
        print(f"Addestramento episodio {episode + 1}: {result}")