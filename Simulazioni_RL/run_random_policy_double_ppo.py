# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:34:39 2022

@author: Daniele
"""


import gym
from gym.envs.registration import register
from stable_baselines3 import PPO
from simglucose.simulation.scenario import CustomScenario
import numpy as np
import pandas as pd
import os
from statistics import mean, stdev
from datetime import datetime
from random import randrange
from datetime import timedelta
import json
import scipy
import time
import math
import warnings
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")


os.chdir('C:\\Users\\utente\\Documents\\GitHub\\simglucose\\Simulazioni_RL')
cwd = os.getcwd()


# PARAMETRI DA SETTARE

patient_type = 'adult'  # 'adolescent'

reward_type = 'new'  # 'magni'

# training parameters

training_learning_rate = '00003'
training_n_step_list = [1024]
training_total_timesteps_list = [1024]

# test parameters

n_days = 10
n_hours = n_days*24
test_timesteps = 2400  # 5 giorni
start_time = datetime.strptime('3/4/2022 12:00 AM', '%m/%d/%Y %I:%M %p')
seed = 42
ma = 1
# ma = 15
ripetizioni = 50


if reward_type == 'new':

    model_path = os.path.join(cwd, 'modelli')

    print('using new function')

    def new_func(x):
        return -0.0417 * x**2 + 10.4167 * x - 525.0017

    def new_reward(BG_last_hour):
        return new_func(BG_last_hour[-1])


elif reward_type == 'magni':

    model_path = os.path.join(cwd, 'modelli_magni')

    print('using magni function')

    def clip_0_15_5(x):
        return min(15.5, max(0, x))

    def magni_risk(b):
        c0 = 1.509
        c1 = 1.084
        c2 = 5.381

        if b < 70:
            return -1.0
        else:
            # exponent = c1 - c2
            inner = (c0 * ((math.log(b))**c1 - c2))
            clipped_value = clip_0_15_5(10 * (inner**2))
            return 1 - (clipped_value / 7.75)

    def new_reward(BG_last_hour):
        b = BG_last_hour[-1]
        return magni_risk(b)


# def quad_func(a,x):
#     return -a*(x-90)*(x-150)

# def quad_reward(BG_last_hour):
#     return quad_func(0.0417, BG_last_hour[-1])


# exp. function
# def exp_func(x,a=0.0417,k=0.3,hypo_treshold = 80, hyper_threshold = 180, exp_bool=True):
#   if exp_bool:
#     return -a*(x-hypo_treshold)*(x-hyper_threshold) - np.exp(-k*(x-hypo_treshold))
#   else:
#     return -a*(x-hypo_treshold)*(x-hyper_threshold)

# def exp_reward(BG_last_hour,a=0.0417,k=0.3,hypo_treshold = 90, hyper_threshold = 150, exp_bool=True):
#     return exp_func(BG_last_hour[-1])


def risk_index(BG, horizon):
    # BG is in mg/dL
    # horizon in samples
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        BG_to_compute = BG[-horizon:]
        fBG = 1.509 * (np.log(BG_to_compute)**1.084 - 5.381)
        rl = 10 * fBG[fBG < 0]**2
        rh = 10 * fBG[fBG > 0]**2
        LBGI = np.nan_to_num(np.mean(rl))
        HBGI = np.nan_to_num(np.mean(rh))
        RI = LBGI + HBGI
    return (LBGI, HBGI, RI)


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

    media = sum(valori) / n
    somma_deviazioni_quadrate = 0.0

    for valore in valori:
        deviazione = valore - media
        somma_deviazioni_quadrate += deviazione ** 2

    varianza = somma_deviazioni_quadrate / n
    deviazione_standard_media = math.sqrt(varianza) / math.sqrt(n)

    return deviazione_standard_media


def insert_dot(string, index):
    return string[:index] + '.' + string[index:]

# simglucose parameters


cgm_name = 'Dexcom'
insulin_pump_name = 'Nuovo'
animate = True
parallel = True


strategy_path = os.path.join(cwd, 'Strategy')
if not os.path.exists(strategy_path):
    os.makedirs(strategy_path)


scenario_usato = '5_days_1000_times'

with open(os.path.join(strategy_path, 'scenarios_'+scenario_usato+'.json')) as json_file:
    scenarios = json.load(json_file)


if patient_type == 'adult' and reward_type == 'new':

    opt_dict = {
        'adult#001': ('014', '014', 160, 90),
        'adult#002': ('014', '014', 160, 90),
        'adult#003': ('014', '014', 160, 90),
        'adult#004': ('014', '014', 160, 90),
        'adult#005': ('014', '014', 160, 90),
        'adult#006': ('014', '014', 160, 90),
        'adult#007': ('014', '014', 160, 90),
        'adult#008': ('014', '014', 160, 90),
        'adult#009': ('014', '014', 160, 90),
        'adult#010': ('014', '014', 160, 90)
    }


elif patient_type == 'adult' and reward_type == 'magni':

    opt_dict = {
        'adult#001': ('012', '007', 160, 90),
        'adult#002': ('014', '006', 165, 90),
        'adult#003': ('009', '008', 170, 90),
        'adult#004': ('010', '007', 160, 90),
        'adult#005': ('012', '008', 160, 90),
        'adult#006': ('012', '008', 165, 90),
        'adult#007': ('011', '006', 165, 90),
        'adult#008': ('011', '006', 160, 90),
        'adult#009': ('012', '005', 170, 90),
        'adult#010': ('009', '008', 165, 90)
    }


elif patient_type == 'adolescent' and reward_type == 'new':

    opt_dict = {
        patient_type+'#001': ('012', '006', 170, 90),
        patient_type+'#002': ('011', '006', 160, 90),
        patient_type+'#003': ('012', '005', 170, 90),
        patient_type+'#004': ('011', '007', 165, 90),
        patient_type+'#005': ('009', '007', 165, 90),
        patient_type+'#006': ('009', '006', 170, 90),
        patient_type+'#007': ('009', '006', 160, 90),
        patient_type+'#008': ('011', '005', 170, 90),
        patient_type+'#009': ('012', '006', 170, 90),
        patient_type+'#010': ('009', '005', 165, 90)
    }

elif patient_type == 'adolescent' and reward_type == 'magni':

    opt_dict = {
        patient_type+'#001': ('010', '006', 165, 90),
        patient_type+'#002': ('009', '007', 170, 90),
        patient_type+'#003': ('011', '005', 160, 90),
        patient_type+'#004': ('009', '007', 165, 90),
        patient_type+'#005': ('009', '006', 170, 90),
        patient_type+'#006': ('010', '006', 160, 90),
        patient_type+'#007': ('009', '008', 165, 90),
        patient_type+'#008': ('014', '007', 165, 90),
        patient_type+'#009': ('009', '005', 160, 90),
        patient_type+'#010': ('009', '006', 170, 90)
    }


for training_n_steps, training_total_timesteps in zip(training_n_step_list, training_total_timesteps_list):

    tir_mean_dict = {
        'paziente': [],
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
        'reward mean': [],
        'reward st dev': [],
        'LBGI mean of means': [],
        'LBGI mean of std': [],
        'HBGI mean of means': [],
        'HBGI mean of std': [],
        'RI mean of means': [],
        'RI mean of std': [],
        'cap iper mean': [],
        'cap ipo mean': [],
        'soglia iper mean': [],
        'soglia ipo mean': [],
        'ripetizioni': [],
        'training learning rate': [],
        'training n steps': [],
        'training timesteps': [],
        'test timesteps': [],
        'scenario': [],
        'start time': []
    }

    for k, v in opt_dict.items():

        paziente = k
        iper = v[0]
        ipo = v[1]
        iper_s = v[2]
        ipo_s = v[3]

        dizionario = {'paziente': paziente,
                      'ins_max': iper}

        df_cap = pd.DataFrame(dizionario, index=[0])
        df_cap['timesteps'] = training_n_steps
        df_cap['target timesteps'] = training_total_timesteps
        df_cap['elapsed_time'] = 0
        df_cap['pazient type'] = patient_type
        df_cap['reward type'] = reward_type
        df_cap['ppo conf'] = 'double'
        df_cap['check_learning'] = 'No'
        df_cap.to_excel(os.path.join(
            strategy_path, 'paz_cap.xlsx'), index=False)

        tempistiche = []

        with pd.ExcelWriter(os.path.join(cwd, 'Risultati', 'performance_'+reward_type+'_reward_'+patient_type+'_random_policy_double_ppo_test_'+paziente+'_timesteps_'+str(test_timesteps)+'_training_nsteps_'+str(training_n_steps)+'_training_tmstp_'+str(training_total_timesteps)+'_ripetizioni_'+str(ripetizioni)+'.xlsx')) as writer:

            print(paziente, iper, ipo, iper_s, ipo_s)

            tir_dict = {'death hypo': [],
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
                        'reward': [],
                        'LBGI mean': [],
                        'LBGI std': [],
                        'HBGI mean': [],
                        'HBGI std': [],
                        'RI mean': [],
                        'RI std': [],
                        'cap iper': [],
                        'cap ipo': [],
                        'soglia iper': [],
                        'soglia ipo': [],
                        'training learning rate': [],
                        'training n steps': [],
                        'training timesteps': [],
                        'test timesteps': [],
                        'ripetizione': [],
                        'scenario': [],
                        'tempo esecuzione': [],
                        }

            for i, scen in zip(range(ripetizioni), scenarios.values()):

                df_cap['ripetizione'] = i
                df_cap.to_excel(os.path.join(
                    strategy_path, 'paz_cap.xlsx'), index=False)
                inizio = time.time()

                lista_BG = []

                # start_time = random_date(d1, d2)
                # scen_long = create_scenario(n_days)
                # scenario = CustomScenario(start_time=start_time, scenario=scen_long)#, seed=seed)
                scen = [tuple(x) for x in scen]
                scenario = CustomScenario(start_time=start_time, scenario=scen)

                # registrazione per train singolo
                register(
                    # id='simglucose-adolescent2-v0',
                    id='simglucose-'+patient_type+'2-v0',
                    # entry_point='simglucose.envs:T1DSimEnv',
                    entry_point='simglucose.envs:PPOSimEnv',
                    kwargs={'patient_name': paziente,
                            'reward_fun': new_reward,
                            'custom_scenario': scenario})

                # if patient_type == 'adult' and reward_type == 'new':
                #     model_ppo_iper = PPO.load(os.path.join(model_path, 'ppo_withcaps_'+paziente+'_nsteps_'+str(training_n_steps)+'_total_tmstp_'+str(training_total_timesteps)+'_lr_00003_insmax'+iper)) # iper
                #     model_ppo_ipo = PPO.load(os.path.join(model_path, 'ppo_withcaps_'+paziente+'_nsteps_'+str(training_n_steps)+'_total_tmstp_'+str(training_total_timesteps)+'_lr_00003_insmax'+ipo))  # ipo

                # elif patient_type == 'adult' and reward_type == 'magni':
                #     model_ppo_iper = PPO.load(os.path.join(model_path, 'magni_ppo_withcaps_'+paziente+'_nsteps_'+str(training_n_steps)+'_total_tmstp_'+str(training_total_timesteps)+'_lr_00003_insmax'+iper)) # iper
                #     model_ppo_ipo = PPO.load(os.path.join(model_path, 'magni_ppo_withcaps_'+paziente+'_nsteps_'+str(training_n_steps)+'_total_tmstp_'+str(training_total_timesteps)+'_lr_00003_insmax'+ipo))  # ipo

                # elif patient_type == 'adolescent':
                #     model_ppo_iper = PPO.load(os.path.join(model_path, "ppo_"+reward_type+"_reward_withcaps_"+paziente+'_nsteps_'+str(training_n_steps)+'_total_tmstp_'+str(training_total_timesteps)+'_lr_00003_insmax'+iper)) # iper
                #     model_ppo_ipo = PPO.load(os.path.join(model_path, "ppo_"+reward_type+"_reward_withcaps_"+paziente+'_nsteps_'+str(training_n_steps)+'_total_tmstp_'+str(training_total_timesteps)+'_lr_00003_insmax'+ipo)) # ipo

                # model_ppo_iper = PPO.load(os.path.join(model_path, "ppo_offline_"+paziente+'_nsteps_'+str(training_n_steps)+'_total_tmstp_'+str(training_total_timesteps)+"_lr_"+training_learning_rate+'_insmax'+iper)) # iper
                # model_ppo_ipo = PPO.load(os.path.join(model_path, "ppo_offline_"+paziente+'_nsteps_'+str(training_n_steps)+'_total_tmstp_'+str(training_total_timesteps)+"_lr_"+training_learning_rate+'_insmax'+ipo))  # ipo

                # model_ppo_iper = PPO.load(os.path.join(model_path, 'ppo_sim_mod_food_hour_'+paziente+'_tmstp'+str(training)+'_lr00003_insmax'+iper+'_customscen')) # iper
                # model_ppo_ipo = PPO.load(os.path.join(model_path, 'ppo_sim_mod_food_hour_'+paziente+'_tmstp'+str(training)+'_lr00003_insmax'+ipo+'_customscen'))  # ipo

                env = gym.make('simglucose-'+patient_type+'2-v0')

                observation = env.reset()

                cgm_list = list()
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

                # ogni timestep equivale a 3 minuti
                for t in range(test_timesteps):

                    # env.render(mode='human')

                    print(observation)

                    lista_BG.append(observation[0][0])

                    float_iper = int(iper) / 100
                    float_ipo = int(ipo) / 100

                    if observation[0][0] < ipo_s:
                        action = 0.0
                    elif observation[0][0] > iper_s:
                        action = round(random.uniform(
                            0, float_iper), 4)  # iper control
                    else:
                        action = round(random.uniform(
                            0, float_ipo), 4)  # ipo control

                    observation, reward, done, info = env.step(action)

                    if observation[0][0] <= 21:
                        counter_death_hypo += 1
                    elif 21 < observation[0][0] < 30:
                        counter_under_30 += 1
                    elif 30 <= observation[0][0] < 40:
                        counter_under_40 += 1
                    elif 40 <= observation[0][0] < 50:
                        counter_under_50 += 1
                    elif 50 <= observation[0][0] < 70:
                        counter_under_70 += 1
                    elif 70 <= observation[0][0] <= 180:
                        counter_euglycem += 1
                    elif 180 < observation[0][0] <= 250:
                        counter_over_180 += 1
                    elif 250 < observation[0][0] <= 400:
                        counter_over_250 += 1
                    elif 400 < observation[0][0] <= 500:
                        counter_over_400 += 1
                    elif 500 < observation[0][0] < 595:
                        counter_over_500 += 1
                    elif observation[0][0] >= 595:
                        counter_death_hyper += 1

                    counter_total += 1
                    print(paziente)
                    print(i+1)
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

                fine = time.time()
                tempo_impiegato = fine - inizio

                tempistiche.append(tempo_impiegato)

                LBGI_mean, HBGI_mean, RI_mean, LBGI_std, HBGI_std, RI_std = risk_index_mod(
                    lista_BG, len(lista_BG))

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
                tir_dict['reward'].append(reward)
                tir_dict['LBGI mean'].append(LBGI_mean)
                tir_dict['LBGI std'].append(LBGI_std)
                tir_dict['HBGI mean'].append(HBGI_mean)
                tir_dict['HBGI std'].append(HBGI_std)
                tir_dict['RI mean'].append(RI_mean)
                tir_dict['RI std'].append(RI_std)
                iper_mod = insert_dot(iper, 1)
                tir_dict['cap iper'].append(iper_mod)
                ipo_mod = insert_dot(ipo, 1)
                tir_dict['cap ipo'].append(ipo_mod)
                tir_dict['soglia iper'].append(iper_s)
                tir_dict['soglia ipo'].append(ipo_s)
                tir_dict['test timesteps'].append(test_timesteps)
                tir_dict['ripetizione'].append(i+1)
                tir_dict['training learning rate'].append(
                    training_learning_rate)
                tir_dict['training n steps'].append(training_n_steps)
                tir_dict['training timesteps'].append(training_total_timesteps)
                tir_dict['scenario'].append(scen)
                tir_dict['tempo esecuzione'].append(tempo_impiegato)

                df_res = pd.DataFrame(tir_dict)

                df_res.to_excel(writer, sheet_name=paziente, index=False)

            tir_mean_dict['paziente'].append(k)
            tir_mean_dict['death hypo mean'].append(
                mean(tir_dict['death hypo']))
            tir_mean_dict['death hypo st dev'].append(
                stdev(tir_dict['death hypo']))
            tir_mean_dict['ultra hypo mean'].append(
                mean(tir_dict['ultra hypo']))
            tir_mean_dict['ultra hypo st dev'].append(
                stdev(tir_dict['ultra hypo']))
            tir_mean_dict['heavy hypo mean'].append(
                mean(tir_dict['heavy hypo']))
            tir_mean_dict['heavy hypo st dev'].append(
                stdev(tir_dict['heavy hypo']))
            tir_mean_dict['severe hypo mean'].append(
                mean(tir_dict['severe hypo']))
            tir_mean_dict['severe hypo st dev'].append(
                stdev(tir_dict['severe hypo']))
            tir_mean_dict['hypo mean'].append(mean(tir_dict['hypo']))
            tir_mean_dict['hypo st dev'].append(stdev(tir_dict['hypo']))
            tir_mean_dict['time in range mean'].append(
                mean(tir_dict['time in range']))
            tir_mean_dict['time in range st dev'].append(
                stdev(tir_dict['time in range']))
            tir_mean_dict['hyper mean'].append(mean(tir_dict['hyper']))
            tir_mean_dict['hyper st dev'].append(stdev(tir_dict['hyper']))
            tir_mean_dict['severe hyper mean'].append(
                mean(tir_dict['severe hyper']))
            tir_mean_dict['severe hyper st dev'].append(
                stdev(tir_dict['severe hyper']))
            tir_mean_dict['heavy hyper mean'].append(
                mean(tir_dict['heavy hyper']))
            tir_mean_dict['heavy hyper st dev'].append(
                stdev(tir_dict['heavy hyper']))
            tir_mean_dict['ultra hyper mean'].append(
                mean(tir_dict['ultra hyper']))
            tir_mean_dict['ultra hyper st dev'].append(
                stdev(tir_dict['ultra hyper']))
            tir_mean_dict['death hyper mean'].append(
                mean(tir_dict['death hyper']))
            tir_mean_dict['death hyper st dev'].append(
                stdev(tir_dict['death hyper']))
            tir_mean_dict['reward mean'].append(mean(tir_dict['reward']))
            tir_mean_dict['reward st dev'].append(stdev(tir_dict['reward']))
            tir_mean_dict['LBGI mean of means'].append(
                mean(tir_dict['LBGI mean']))
            tir_mean_dict['LBGI mean of std'].append(
                mean_std(tir_dict['LBGI std']))
            tir_mean_dict['HBGI mean of means'].append(
                mean(tir_dict['HBGI mean']))
            tir_mean_dict['HBGI mean of std'].append(
                mean_std(tir_dict['HBGI std']))
            tir_mean_dict['RI mean of means'].append(mean(tir_dict['RI mean']))
            tir_mean_dict['RI mean of std'].append(
                mean_std(tir_dict['RI std']))
            tir_mean_dict['cap iper mean'].append(tir_dict['cap iper'][0])
            tir_mean_dict['cap ipo mean'].append(tir_dict['cap ipo'][0])
            tir_mean_dict['soglia iper mean'].append(
                tir_dict['soglia iper'][0])
            tir_mean_dict['soglia ipo mean'].append(tir_dict['soglia ipo'][0])
            tir_mean_dict['test timesteps'].append(
                tir_dict['test timesteps'][0])
            tir_mean_dict['ripetizioni'].append(ripetizioni)
            tir_mean_dict['training learning rate'].append(
                training_learning_rate)
            tir_mean_dict['training n steps'].append(training_n_steps)
            tir_mean_dict['training timesteps'].append(
                training_total_timesteps)
            tir_mean_dict['scenario'].append(scenario_usato)
            tir_mean_dict['start time'].append(start_time)

            df_res_mean = pd.DataFrame(tir_mean_dict)

            df_res_mean.to_excel(writer, sheet_name='risultati', index=False)
