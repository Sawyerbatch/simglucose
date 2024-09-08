# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 23:52:14 2024

@author: Daniele
"""

from __future__ import annotations
import itertools
import shutil
import statistics
import scipy
import openpyxl
import csv
import glob
import os
import time
from typing import Optional
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import numpy as np
from pettingzoo.sisl import waterworld_v4
import pandas as pd
import gymnasium
import time
import warnings
import json
from statistics import mean, stdev
from datetime import datetime
from stable_baselines3 import PPO
from simglucose.envs import T1DSimGymnasiumEnv_MARL
from simglucose.simulation.scenario import CustomScenario
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from pettingzoo.utils import wrappers
import pettingzoo
import gymnasium as gym
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env

# Disable all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Disabilitare specifici avvisi basati sul contenuto del messaggio
warnings.filterwarnings("ignore")

# Allora Morty, la lezione è finita. Abbiamo degli affari da sbrigare a pochi minuti luce, a sud, di qui.


# def new_func(x):
#     return -0.0417 * x**2 + 10.4167 * x - 525.0017

def new_func(x):
    return -(x - 110) * (x - 140)


def new_reward(BG_last_hour):
    return new_func(BG_last_hour[-1])


def create_scenario(n_days, cho_daily=230):

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
            scenario.append((hour_break, cho_break))
        if int(np.random.randint(100)) < 100:
            scenario.append((hour_lunch, cho_lunch))
        if int(np.random.randint(100)) < 30:
            scenario.append((hour_snack, cho_snack))
        if int(np.random.randint(100)) < 95:
            scenario.append((hour_dinner, cho_dinner))
        if int(np.random.randint(100)) < 3:
            scenario.append((hour_night, cho_night))

        # cho_sum += cho_break + cho_lunch + cho_snack + cho_dinner + cho_night

    return scenario

# # Generate 200 scenarios, each with 30 days
# scenarios = {str(i): create_scenario(30) for i in range(200)}

# # Save the scenarios to a JSON file
# with open('scenarios_30_days_200_times.json', 'w') as f:
#     json.dump(scenarios, f)


def risk_index_mod(BG, horizon):
    # BG is in mg/dL
    # horizon in samples
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        BG_to_compute = BG[-horizon:]
        fBG = 1.509 * (np.log(BG_to_compute)**1.084 - 5.381)
        rl = 10 * fBG[fBG < 0]**2
        rh = 10 * fBG[fBG > 0]**2
        LBGI_mean = np.nan_to_num(np.mean(rl))
        HBGI_mean = np.nan_to_num(np.mean(rh))
        RI_mean = LBGI_mean + HBGI_mean
        LBGI_std = np.nan_to_num(np.std(rl))
        HBGI_std = np.nan_to_num(np.std(rh))
        # Propagation of uncertainty
        RI_std = np.sqrt(LBGI_std**2 + HBGI_std**2)
    return (LBGI_mean, HBGI_mean, RI_mean, LBGI_std, HBGI_std, RI_std)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def mean_std(valori):
    n = len(valori)
    if n == 0:
        return 0.0


class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper):
    """Wrapper to allow PettingZoo environments to be used with SB3 illegal action masking."""

    def reset(self, seed=None, options=None):
        """Gymnasium-like reset function which assigns obs/action spaces to be the same for each agent.

        This is required as SB3 is designed for single-agent RL and doesn't expect obs/action spaces to be functions
        """
        super().reset(seed, options)

        # Strip the action mask out from the observation space
        self.observation_space = super().observation_space(self.possible_agents[0])[
            "observation"
        ]
        self.action_space = super().action_space(self.possible_agents[0])

        # Return initial observation, info (PettingZoo AEC envs do not by default)
        return self.observe(self.agent_selection), {}

    def step(self, action):
        """Gymnasium-like step function, returning observation, reward, termination, truncation, info."""
        super().step(action)
        return super().last()

    def observe(self, agent):
        """Return only raw observation, removing action mask."""
        return super().observe(agent)["observation"]

    def action_mask(self):
        """Separate function used in order to access the action mask."""
        return super().observe(self.agent_selection)["action_mask"]


def mask_fn(env):
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_mask()


def train_action_mask(env_fn, folder, paziente, train_timesteps, n_steps,
                      learning_rate, batch_size, n_epochs, gamma,
                      gae_lambda, clip_range, ent_coef, vf_coef,
                      max_grad_norm, target_kl, seed=0, **env_kwargs):
    """Train a single model to play as each agent in a zero-sum game environment using invalid action masking."""
    # env = env_fn.env(**env_kwargs)

    print(
        f"Starting training on patient {paziente} with {str(env_fn.metadata['name'])}.")

    # env_fn = make_vec_env(env_fn, n_envs=4)

    # Custom wrapper to convert PettingZoo envs to work with SB3 action masking
    env = SB3ActionMaskWrapper(env_fn)

    env.reset(seed=seed)  # Must call reset() in order to re-define the spaces

    env = ActionMasker(env, mask_fn)  # Wrap to enable masking (SB3 function)
    # MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
    # with ActionMasker. If the wrapper is detected, the masks are automatically
    # retrieved and used when learning. Note that MaskablePPO does not accept
    # a new action_mask_fn kwarg, as it did in an earlier draft.

    # batch_size = 240

    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1,
                        n_steps=n_steps, learning_rate=learning_rate,
                        batch_size=batch_size, n_epochs=n_epochs, gamma=gamma,
                        gae_lambda=gae_lambda, clip_range=clip_range, ent_coef=ent_coef,
                        vf_coef=vf_coef, max_grad_norm=max_grad_norm, target_kl=target_kl
                        )
    model.set_random_seed(seed)
    model.learn(total_timesteps=train_timesteps, progress_bar=True,
                reset_num_timesteps=True)

    model.save(os.path.join(folder,
                            f"{env.unwrapped.metadata.get('name')}_{paziente}_{train_timesteps}_{time_suffix_Min}"))

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()


def eval_action_mask(paziente, scenarios, tir_mean_dict, time_suffix, test_folder,
                     num_tests, learning_rate, batch_size, n_epochs, gamma, gae_lambda, clip_range, ent_coef,
                     vf_coef, max_grad_norm, target_kl, test_timesteps,
                     render_mode=None, last_models=False, morty_cap=7, rick_cap=15,
                     soglia_ipo=85, soglia_iper=120, **env_kwargs):

    if last_models:
        for m in os.listdir(os.path.join(main_train_folder, 'Tr_'+p, f"Training_{p}_{learning_rate}_{batch_size}_{n_epochs}_{gamma}_{gae_lambda}_{clip_range}_{ent_coef}_{vf_coef}_{max_grad_norm}_{target_kl}")):
            if m.startswith('T1DSimGymnasiumEnv_MARL_'+paziente):
                model = MaskablePPO.load(os.path.join(
                    main_train_folder, 'Tr_' +
                    p, f"Training_{p}_{learning_rate}_{batch_size}_{n_epochs}_{gamma}_{gae_lambda}_{clip_range}_{ent_coef}_{vf_coef}_{max_grad_norm}_{target_kl}",
                    m.split('.')[0]))

    else:
        for m in os.listdir('Models'):
            if m.startswith('T1DSimGymnasiumEnv_MARL_'+paziente):
                model = MaskablePPO.load('Models\\'+m.split('.')[0])

    print(model)

    round_rewards = []

    tir_dict = {'ripetizione': [],
                'death hypo': [],
                'ultra hypo': [],
                'heavy hypo': [],
                'severe hypo': [],
                'hypo': [],
                'time in range': [],
                'hyper': [],
                'severe hyper': [],
                'heavy hyper': [],
                'ultra hyper': [],
                'death hyper': [],
                'HBGI mean': [],
                'HBGI std': [],
                'LBGI mean': [],
                'LBGI std': [],
                'RI mean': [],
                'RI std': [],
                'Rick reward': [],
                'Morty reward': [],
                'Jerry reward': [],
                'test timesteps': [],
                'soglia_ipo': [],
                'soglia_iper': [],
                'Morty_cap': [],
                'Rick_cap': [],
                'scenario': [],
                }

    if True:
        # with pd.ExcelWriter(os.path.join(test_patient_folder, f'history_{paziente}_{soglia_ipo}_{soglia_iper}_{morty_cap}_{rick_cap}_{time_suffix_Min}.xlsx')) as patient_writer:

        for i, scen in zip(range(1, num_tests+1), scenarios.values()):
            # print(scen)
            lista_BG = []

            sheet_name = f'Game_{i}'

            scen = [tuple(x) for x in scen]
            test_scenario = CustomScenario(
                start_time=start_time, scenario=scen)

            def create_env(**kwargs):
                env = T1DSimGymnasiumEnv_MARL(**kwargs)
                # Additional wrappers can be added here if needed
                return env

            env_fn = lambda **kwargs: create_env(
                patient_name=paziente,
                custom_scenario=test_scenario,
                reward_fun=new_reward,
                # seed=123,
                render_mode="human",
                training=False,
                morty_cap=morty_cap,
                rick_cap=rick_cap,
                soglia_ipo=soglia_ipo,
                soglia_iper=soglia_iper,
            )

            env = env_fn(render_mode=render_mode, **env_kwargs)

            env.reset(seed=i)
            env.action_space(env.possible_agents[0]).seed(i)

            timestep = 0

            # scores = {agent: 0 for agent in env.possible_agents}
            # total_rewards = {agent: 0 for agent in env.possible_agents}

            obs = env.reset()  # Resetta l'ambiente e ottieni l'osservazione iniziale
            # print(obs)

            data_list = []

            # cgm_list = list()
            counter_death_hypo = 0
            counter_under_30 = 0
            counter_under_40 = 0
            counter_under_50 = 0
            counter_under_70 = 0
            counter_euglycem = 0
            counter_over_180 = 0
            counter_over_250 = 0
            counter_over_400 = 0
            counter_over_500 = 0
            counter_death_hyper = 0

            counter_total = 0

            tir = np.zeros(shape=(11,))

            truncation = False
            termination = False

            while timestep < test_timesteps:

                # print('timesteeeep', timestep)

                for agent in env.agent_iter():
                    # print('agent_iter:',agent)
                    obs, reward, _, _, info = env.last()
                    # if termination:
                    #     print('termination == done')
                    # elif truncation:
                    #     print('truncation == done')
                    observations, action_mask = obs.values()
                    observation = observations[0]

                    print('observation', observation)

                    if observation < 20 or observation > 600:
                        truncation = True

                    if timestep >= test_timesteps:
                        termination = True

                    if observation <= 21:
                        counter_death_hypo += 1
                    elif 21 < observation < 30:
                        counter_under_30 += 1
                    elif 30 <= observation < 40:
                        counter_under_40 += 1
                    elif 40 <= observation < 50:
                        counter_under_50 += 1
                    elif 50 <= observation < 70:
                        counter_under_70 += 1
                    elif 70 <= observation <= 180:
                        counter_euglycem += 1
                    elif 180 < observation <= 250:
                        counter_over_180 += 1
                    elif 250 < observation <= 400:
                        counter_over_250 += 1
                    elif 400 < observation <= 500:
                        counter_over_400 += 1
                    elif 500 < observation < 595:
                        counter_over_500 += 1
                    elif observation >= 595:
                        counter_death_hyper += 1

                    counter_total += 1
                    # print(paziente)
                    # print(i+1)

                    print('Patient ', paziente)
                    print('Results ', sheet_name, ' Timestep_'+str(timestep))
                    print('Active agent: ', env.agent_selection)
                    tir[0] = (counter_death_hypo/counter_total)*100
                    print('death_hypo:', tir[0])
                    tir[1] = (counter_under_30 / counter_total) * 100
                    print('under_30:', tir[1])
                    tir[2] = (counter_under_40 / counter_total) * 100
                    print('under_40:', tir[2])
                    tir[3] = (counter_under_50 / counter_total) * 100
                    print('under_50:', tir[3])
                    tir[4] = (counter_under_70 / counter_total) * 100
                    print('under_70:', tir[4])
                    tir[5] = (counter_euglycem / counter_total) * 100
                    print('euglycem:', tir[5])
                    tir[6] = (counter_over_180 / counter_total) * 100
                    print('over_180:', tir[6])
                    tir[7] = (counter_over_250 / counter_total) * 100
                    print('over_250:', tir[7])
                    tir[8] = (counter_over_400 / counter_total) * 100
                    print('over_400:', tir[8])
                    tir[9] = (counter_over_500 / counter_total) * 100
                    print('over_500:', tir[9])
                    tir[10] = (counter_death_hyper / counter_total) * 100
                    print('death_hyper:', tir[10])
                    print()
                    lista_BG.append(observation)

                    if timestep > 1:

                        print('INS:', info['insulin'])

                        # data_list.append({
                        #         'Timestep': timestep,
                        #         'CGM': round(env.obs.CGM, 3),
                        #         'dCGM': round(env.obs.dCGM, 3),
                        #         'IOB': round(env.obs.IOB, 3),
                        #         'h_zone': env.obs.h_zone,
                        #         'food': env.obs.food,
                        #         'BG': info['bg'],
                        #         'LBGI': info['lbgi'],
                        #         'HBGI': info['hbgi'],
                        #         'RISK': info['risk'],
                        #         'INS': info['insulin'],
                        #         'CHO': info['meal'],
                        #         'Active_agent': agent,
                        #         'Rick_Reward': str(round(env.rewards['Rick'],3)),
                        #         'Morty_Reward': str(round(env.rewards['Morty'],3)),
                        #         'Jerry_Reward': str(round(env.rewards['Jerry'],3)),
                        #         'Truncation': truncation,
                        #         'Termination': termination,
                        #         'Soglia_ipo': soglia_ipo,
                        #         'Soglia_iper': soglia_iper,
                        #         'Morty_cap': morty_cap,
                        #         'Rick_cap': rick_cap

                        #     })

                        # df = pd.DataFrame(data_list)

                        # df.to_excel(patient_writer, sheet_name=sheet_name, index=False)

                        # df_hist = env.show_history()

                    if truncation:
                        print(truncation)

                        # truncation = True
                    # if termination or truncation:
                    #     if env.rewards[env.possible_agents[0]] != env.rewards[env.possible_agents[1]]:
                    #         winner = max(env.rewards, key=env.rewards.get)
                    #         scores[winner] += env.rewards[winner]
                    #     for a in env.possible_agents:
                    #         total_rewards[a] += env.rewards[a]
                    #     round_rewards.append(env.rewards)
                        # break

                    else:
                        # print('possible_agent', env.possible_agents)
                        # if agent == env.possible_agents[0]:
                        # print('ageeeent', agent)
                        # act = env.action_space(agent).sample(action_mask)
                        # else:
                        # print(observations.shape)
                        act = int(model.predict(
                            [observations], action_masks=action_mask, deterministic=False)[0])
                        # act = int(model.predict([observation], action_masks=action_mask, deterministic=True)[0])
                        env.step(act)
                    timestep += 1
                    if termination:

                        break
                if termination:

                    break

            # Initialize the placeholder filename
            placeholder_filename = os.path.join(
                test_patient_folder, f'results_{paziente}_{learning_rate}_{batch_size}_{n_epochs}_{gamma}_{gae_lambda}_{clip_range}_{ent_coef}_{vf_coef}_{max_grad_norm}_{target_kl}_{time_suffix_Min}.xlsx')

            # Open the ExcelWriter with the placeholder filename
            with pd.ExcelWriter(placeholder_filename) as middle_writer:

                LBGI_mean, HBGI_mean, RI_mean, LBGI_std, HBGI_std, RI_std = risk_index_mod(
                    lista_BG, len(lista_BG))

                # avg_reward = sum(total_rewards.values()) / len(total_rewards.values())
                # print("Total rewards:", total_rewards)
                # print(f"Average reward: {avg_reward}")

                tir_dict['ripetizione'].append(i)
                tir_dict['death hypo'].append(tir[0])
                tir_dict['ultra hypo'].append(tir[1])
                tir_dict['heavy hypo'].append(tir[2])
                tir_dict['severe hypo'].append(tir[3])
                tir_dict['hypo'].append(tir[4])
                tir_dict['time in range'].append(tir[5])
                tir_dict['hyper'].append(tir[6])
                tir_dict['severe hyper'].append(tir[7])
                tir_dict['heavy hyper'].append(tir[8])
                tir_dict['ultra hyper'].append(tir[9])
                tir_dict['death hyper'].append(tir[10])
                tir_dict['HBGI mean'].append(HBGI_mean)
                tir_dict['HBGI std'].append(HBGI_std)
                tir_dict['LBGI mean'].append(LBGI_mean)
                tir_dict['LBGI std'].append(LBGI_std)
                tir_dict['RI mean'].append(RI_mean)
                tir_dict['RI std'].append(RI_std)
                tir_dict['Rick reward'].append(
                    str(round(env.rewards['Rick'], 3)))
                tir_dict['Morty reward'].append(
                    str(round(env.rewards['Morty'], 3)))
                tir_dict['Jerry reward'].append(
                    str(round(env.rewards['Jerry'], 3)))
                tir_dict['test timesteps'].append(timestep-1)
                tir_dict['soglia_ipo'].append(soglia_ipo)
                tir_dict['soglia_iper'].append(soglia_iper)
                tir_dict['Morty_cap'].append(morty_cap)
                tir_dict['Rick_cap'].append(rick_cap)
                tir_dict['scenario'].append(scen)
                # tir_dict['tempo esecuzione'].append(tempo_impiegato)

                df_result = pd.DataFrame(tir_dict)

                df_result.to_excel(
                    middle_writer, sheet_name='general_results', index=False)
                # middle_writer.close()

        # Calculate the mean of the 'time in range' column
        time_in_range_mean = df_result['time in range'].mean()

        # Create a new filename incorporating the mean value
        modified_filename = os.path.join(
            test_patient_folder, f'results_{paziente}_TIRmean_{round(time_in_range_mean, 2)}_{learning_rate}_{batch_size}_{n_epochs}_{gamma}_{gae_lambda}_{clip_range}_{ent_coef}_{vf_coef}_{max_grad_norm}_{target_kl}_{time_suffix_Min}.xlsx')

        # Delete the placeholder file
        if os.path.exists(placeholder_filename):
            os.remove(placeholder_filename)

        # Save the DataFrame to the new filename
        with pd.ExcelWriter(modified_filename) as final_writer:
            df_result.to_excel(
                final_writer, sheet_name='general_results', index=False)

        env.close()

        # rewards = [env.rewards['Jerry'], env.rewards['Morty'], env.rewards['Rick']]

        # return scores, total_rewards, round_rewards
        return env.rewards, tir_dict


# %%
if __name__ == "__main__":

    batch_time_unit = 480

    # total_timesteps_list = [16 * batch_time_unit] #[15360] #[960]
    total_timesteps_list = [batch_time_unit, 4 * batch_time_unit, 16 * batch_time_unit]
    # total_timesteps_list = [64 * batch_time_unit, 64 * batch_time_unit, 64 * batch_time_unit]
    # total_timesteps_list = [batch_time_unit]
    n_timesteps_list = [batch_time_unit, batch_time_unit, batch_time_unit]
    # n_timesteps_list = [16 * batch_time_unit]

    # total_timesteps_list = [960, 7680]  # [15360] #[7680]
    # n_timesteps_list = [480]

    # Learning rate values for optimization
    # learning_rate_list = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3] # Grid Search
    learning_rate_list = [3e-4]  # Default value
    # learning_rate_list = [1e-4]  # New value

    # batch_size_list = [32, 64, 128]  # Size of minibatches during optimization
    # batch_size_list = [128]  # Default value
    batch_size_list = [32]  # New value

    # Number of epochs for optimizing the surrogate loss
    # n_epochs_list = [5, 10, 20] # Grid Search
    n_epochs_list = [10]  # Default value
    # n_epochs_list = [5]  # New value

    # gamma_list = [0.95, 0.99, 0.999]  # Discount factor for future rewards # Grid Search
    # gamma_list = [0.99]  # Default value
    gamma_list = [0.95]  # New value

    # Factor for bias-variance trade-off in GAE
    # gae_lambda_list = [0.90, 0.95, 1.0] # Grid Search
    gae_lambda_list = [1.00]  # Default value
    # gae_lambda_list = [0.90]  # New value

    # Clipping parameter for the policy updates
    # clip_range_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5] #Grid Search
    clip_range_list = [0.2]  # Default value
    # clip_range_list = [0.2]  # New value

    # Entropy coefficient to encourage exploration
    # ent_coef_list = [0.0, 0.01, 0.05, 0.1] # Grid Search
    ent_coef_list = [0.0]  # Default value
    # ent_coef_list = [0.05]  # New value

    # Value function coefficient in the loss function
    # vf_coef_list = [0.25, 0.5, 0.75] # Grid Search
    vf_coef_list = [0.5]  # Default value
    # vf_coef_list = [0.75]  # New value

    # Maximum gradient norm for gradient clipping
    max_grad_norm_list = [0.5, 1.0, 2.0] # Grid Search
    # max_grad_norm_list = [0.5]  # Default value
    # max_grad_norm_list = [2.0]  # New value

    target_kl_list = [0.01, 0.05, 0.1]  # Target KL divergence between updates Grid Search
    # target_kl_list = [None]  # Default value (typically None, meaning no KL target)
    # target_kl_list = [0.01]  # Default value

    num_test = 10
    test_timesteps = 2400
    n_days_scenario = 365

    diz = {
        # soglia_ipo, soglia_iper, morty_cap, rick_cap
        # 'adult#001': [100, 160, 7, 11],
        # 'adult#002': [90, 160, 10, 11],
        # 'adult#003': [90, 140, 8, 10],
        'adult#004': [80, 190, 4, 5],
        # 'adult#005': [80, 180, 10, 12],
        # 'adult#006': [90, 160, 7, 10],
        # 'adult#007': [80, 120, 4, 6],
        # 'adult#008': [90, 140, 6, 10],
        # 'adult#009': [90, 140, 7, 11],
        # 'adult#010': [90, 160, 7, 10],
    }

    # for train_timesteps in [960,2400,4800]:
    for train_timesteps, n_steps in zip(total_timesteps_list, n_timesteps_list):

        last_models = False
        # Ottieni la data corrente
        current_time = datetime.now()

        time_suffix = current_time.strftime("%Y%m%d_%H%M%S")
        time_suffix_Min = current_time.strftime("%Y%m%d_%H%M")

        general_results_path = f'test_general_results_{time_suffix_Min}.xlsx'
        test_folder = f"Test\\Test_{time_suffix_Min}"+"_train_"+str(
            train_timesteps)+"("+str(n_steps)+")_test_"+str(test_timesteps)
        # Crea la cartella se non esiste già
        os.makedirs(test_folder, exist_ok=True)

        main_train_folder = os.path.join("Training",
                                         f"Training_{time_suffix_Min}_train_"+str(train_timesteps)+"("+str(n_steps)+")")

        for learning_rate, batch_size, n_epochs, gamma, gae_lambda, clip_range, ent_coef, vf_coef, max_grad_norm, target_kl in itertools.product(
                learning_rate_list, batch_size_list, n_epochs_list, gamma_list,
                gae_lambda_list, clip_range_list, ent_coef_list, vf_coef_list, max_grad_norm_list, target_kl_list):

            # train_timesteps = 2400
            # n_steps= 1024

            # pazienti = [
            # 'adult#001',
            # 'adult#002',
            # 'adult#003',
            # 'adult#004',
            # 'adult#005',
            # 'adult#006',
            # 'adult#007',
            # 'adult#008',
            # 'adult#009',
            # 'adult#010',
            # ]

            # test fixed scenario
            # with open('scenarios_5_days_1000_times.json') as json_file:
            #     test_scenarios = json.load(json_file)

            with open('scenarios_30_days_200_times.json') as json_file:
                test_scenarios = json.load(json_file)

            tir_mean_dict = {
                'paziente': [],
                'avg reward': [],
                'death hypo mean': [],
                'death hypo st dev': [],
                'ultra hypo mean': [],
                'ultra hypo st dev': [],
                'heavy hypo mean': [],
                'heavy hypo st dev': [],
                'severe hypo mean': [],
                'severe hypo st dev': [],
                'hypo mean': [],
                'hypo st dev': [],
                'time in range mean': [],
                'time in range st dev': [],
                'hyper mean': [],
                'hyper st dev': [],
                'severe hyper mean': [],
                'severe hyper st dev': [],
                'heavy hyper mean': [],
                'heavy hyper st dev': [],
                'ultra hyper mean': [],
                'ultra hyper st dev': [],
                'death hyper mean': [],
                'death hyper st dev': [],
                'HBGI mean of means': [],
                'HBGI mean of std': [],
                'LBGI mean of means': [],
                'LBGI mean of std': [],
                'RI mean of means': [],
                'RI mean of std': [],
                'ripetizioni': [],
                'training steps': [],
                'training time': [],
                'testing time': []

            }

            time_dict = {}

            # for p in (pazienti):
            for p in (list(diz.keys())):

                # Stampa o utilizza la combinazione corrente degli iperparametri
                print(f"Training patient {p} with: "
                      f"train_timesteps={train_timesteps}, n_steps={n_steps}, learning_rate={learning_rate}, "
                      f"batch_size={batch_size}, n_epochs={n_epochs}, gamma={gamma}, gae_lambda={gae_lambda}, "
                      f"clip_range={clip_range}, ent_coef={ent_coef}, vf_coef={vf_coef}, max_grad_norm={max_grad_norm}, "
                      f"target_kl={target_kl}")

                soglia_ipo, soglia_iper, morty_cap, rick_cap = diz[p][0], diz[p][1], diz[p][2], diz[p][3]

                if rick_cap <= morty_cap:
                    continue

                start_training_time = time.time()

                # Crea il nome della cartella usando il suffisso di tempo
                train_folder = os.path.join(
                    main_train_folder, f"Tr_{p}", f"Training_{p}_{learning_rate}_{batch_size}_{n_epochs}_{gamma}_{gae_lambda}_{clip_range}_{ent_coef}_{vf_coef}_{max_grad_norm}_{target_kl}")

                test_patient_folder = os.path.join(test_folder, p)
                os.makedirs(test_patient_folder, exist_ok=True)

                start_time = datetime.strptime(
                    '3/4/2024 12:00 AM', '%m/%d/%Y %I:%M %p')

                # train random scenario
                scen_long = create_scenario(n_days_scenario)
                train_scenario = CustomScenario(
                    start_time=start_time, scenario=scen_long)  # , seed=seed)

                def env(**kwargs):
                    env = T1DSimGymnasiumEnv_MARL(**kwargs)
                    # env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
                    # env = wrappers.AssertOutOfBoundsWrapper(env)
                    # env = wrappers.OrderEnforcingWrapper(env)
                    return env

                env_kwargs = {}

                # TRAIN
                env_fn = env(
                    patient_name=p,
                    custom_scenario=train_scenario,
                    reward_fun=new_reward,
                    # seed=123,
                    render_mode="human",
                    training=True,
                    folder=train_folder,
                    morty_cap=morty_cap,
                    rick_cap=rick_cap,
                    soglia_ipo=soglia_ipo,
                    soglia_iper=soglia_iper,
                )

                train_action_mask(env_fn, train_folder, p,
                                  train_timesteps, n_steps,
                                  learning_rate, batch_size,
                                  n_epochs, gamma,
                                  gae_lambda, clip_range,
                                  ent_coef, vf_coef, max_grad_norm,
                                  target_kl, seed=0, **env_kwargs)
                last_models = True

                # # env_fn = env(
                # #         patient_name=p,
                # #         custom_scenario=train_scenario,
                # #         reward_fun=new_reward,
                # #         # seed=123,
                # #         render_mode="human",
                # #         training = False,
                # #         folder=test_folder
                # #     )

                end_training_time = time.time()

                elapsed_training_time = end_training_time - start_training_time

                start_test_time = time.time()

                rewards, tir_dict = eval_action_mask(p,
                                                     test_scenarios,
                                                     tir_mean_dict,
                                                     time_suffix,
                                                     test_patient_folder,
                                                     num_test,
                                                     learning_rate,
                                                     batch_size,
                                                     n_epochs,
                                                     gamma,
                                                     gae_lambda,
                                                     clip_range,
                                                     ent_coef,
                                                     vf_coef,
                                                     max_grad_norm,
                                                     target_kl,
                                                     test_timesteps,
                                                     last_models=last_models,
                                                     morty_cap=morty_cap,
                                                     rick_cap=rick_cap,
                                                     soglia_ipo=soglia_ipo,
                                                     soglia_iper=soglia_iper,
                                                     render_mode=None,
                                                     **env_kwargs)

                # Note that use of masks is manual and optional outside of learning,
                # so masking can be "removed" at testing time
                # model.predict(observation, action_masks=valid_action_array)

                end_test_time = time.time()

                elapsed_test_time = end_test_time - start_test_time

                # with pd.ExcelWriter(os.path.join(test_folder, general_results_path)) as final_writer:

                #     tir_mean_dict['paziente'].append(p)
                #     avg_reward = statistics.mean(rewards.values())
                #     tir_mean_dict['avg reward'].append(avg_reward)
                #     tir_mean_dict['death hypo mean'].append(mean(tir_dict['death hypo']))
                #     tir_mean_dict['death hypo st dev'].append(stdev(tir_dict['death hypo']))
                #     tir_mean_dict['ultra hypo mean'].append(mean(tir_dict['ultra hypo']))
                #     tir_mean_dict['ultra hypo st dev'].append(stdev(tir_dict['ultra hypo']))
                #     tir_mean_dict['heavy hypo mean'].append(mean(tir_dict['heavy hypo']))
                #     tir_mean_dict['heavy hypo st dev'].append(stdev(tir_dict['heavy hypo']))
                #     tir_mean_dict['severe hypo mean'].append(mean(tir_dict['severe hypo']))
                #     tir_mean_dict['severe hypo st dev'].append(stdev(tir_dict['severe hypo']))
                #     tir_mean_dict['hypo mean'].append(mean(tir_dict['hypo']))
                #     tir_mean_dict['hypo st dev'].append(stdev(tir_dict['hypo']))
                #     tir_mean_dict['time in range mean'].append(mean(tir_dict['time in range']))
                #     tir_mean_dict['time in range st dev'].append(stdev(tir_dict['time in range']))
                #     tir_mean_dict['hyper mean'].append(mean(tir_dict['hyper']))
                #     tir_mean_dict['hyper st dev'].append(stdev(tir_dict['hyper']))
                #     tir_mean_dict['severe hyper mean'].append(mean(tir_dict['severe hyper']))
                #     tir_mean_dict['severe hyper st dev'].append(stdev(tir_dict['severe hyper']))
                #     tir_mean_dict['heavy hyper mean'].append(mean(tir_dict['heavy hyper']))
                #     tir_mean_dict['heavy hyper st dev'].append(stdev(tir_dict['heavy hyper']))
                #     tir_mean_dict['ultra hyper mean'].append(mean(tir_dict['ultra hyper']))
                #     tir_mean_dict['ultra hyper st dev'].append(stdev(tir_dict['ultra hyper']))
                #     tir_mean_dict['death hyper mean'].append(mean(tir_dict['death hyper']))
                #     tir_mean_dict['death hyper st dev'].append(stdev(tir_dict['death hyper']))
                #     tir_mean_dict['LBGI mean of means'].append(mean(tir_dict['LBGI mean']))
                #     tir_mean_dict['LBGI mean of std'].append(mean_std(tir_dict['LBGI std']))
                #     tir_mean_dict['HBGI mean of means'].append(mean(tir_dict['HBGI mean']))
                #     tir_mean_dict['HBGI mean of std'].append(mean_std(tir_dict['HBGI std']))
                #     tir_mean_dict['RI mean of means'].append(mean(tir_dict['RI mean']))
                #     tir_mean_dict['RI mean of std'].append(mean_std(tir_dict['RI std']))
                #     tir_mean_dict['ripetizioni'].append(num_test)
                #     tir_mean_dict['training steps'].append(train_timesteps)
                #     tir_mean_dict['training time'].append(elapsed_training_time)
                #     tir_mean_dict['testing time'].append(elapsed_test_time)

                #     df_cap_mean = pd.DataFrame(tir_mean_dict)

                #     df_cap_mean.to_excel(final_writer, sheet_name='risultati_finali', index=False)
