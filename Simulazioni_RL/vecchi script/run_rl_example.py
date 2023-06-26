from simglucose.simulation.user_interface import simulate
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.random_ctrller import RandomController
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.pid_ctrller import PIDController
from stable_baselines3 import PPO
from simglucose.controller.ppo_ctrller import PPOController
from datetime import datetime
from datetime import timedelta
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.env import T1DSimEnv, PPOSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
import os
import numpy as np
import pandas as pd
import json


# def create_scenario(n_days, cho_daily=230):

#   scenario = []
#   cho_sum = 0
#   mu_break, sigma_break = 8, 3 
#   mu_lunch, sigma_lunch = 13, 1
#   mu_snack, sigma_snack = 17, 2
#   mu_dinner, sigma_dinner = 21, 2
#   mu_night, sigma_night = 24, 2

#   for i in range(n_days):

#     mu_cho_break, sigma_cho_break = cho_daily*0.15, 15 
#     mu_cho_lunch, sigma_cho_lunch = cho_daily*0.45, 45
#     mu_cho_snack, sigma_cho_snack = cho_daily*0.05, 5
#     mu_cho_dinner, sigma_cho_dinner = cho_daily*0.35, 35
#     mu_cho_night, sigma_cho_night = cho_daily*0.05, 5

#     hour_break = int(np.random.normal(mu_break, sigma_break/2)) + 24*i
#     hour_lunch = int(np.random.normal(mu_lunch, sigma_lunch/2)) + 24*i
#     hour_snack = int(np.random.normal(mu_snack, sigma_snack/2)) + 24*i
#     hour_dinner = int(np.random.normal(mu_dinner, sigma_dinner/2)) + 24*i
#     hour_night = int(np.random.normal(mu_night, sigma_night/2)) + 24*i

#     cho_break = int(np.random.normal(mu_cho_break, sigma_cho_break/2))
#     cho_lunch = int(np.random.normal(mu_cho_lunch, sigma_cho_lunch/2))
#     cho_snack = int(np.random.normal(mu_cho_snack, sigma_cho_snack/2))
#     cho_dinner = int(np.random.normal(mu_cho_dinner, sigma_cho_dinner/2))
#     cho_night = int(np.random.normal(mu_cho_night, sigma_cho_night/2))

#     if int(np.random.randint(100)) < 60:
#       scenario.append((hour_break,cho_break))
#     if int(np.random.randint(100)) < 100:
#       scenario.append((hour_lunch,cho_lunch))
#     if int(np.random.randint(100)) < 30:
#       scenario.append((hour_snack,cho_snack))
#     if int(np.random.randint(100)) < 95:
#       scenario.append((hour_dinner,cho_dinner))
#     if int(np.random.randint(100)) < 3:
#       scenario.append((hour_night,cho_night))

#     #cho_sum += cho_break + cho_lunch + cho_snack + cho_dinner + cho_night

#   return scenario


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


scenario_usato = '5_days_1000_times'

with open(os.path.join(model_path, 'Risultati\Strategy', 'scenarios_'+scenario_usato+'.json')) as json_file:
    scenarios = json.load(json_file)


'''
Main user interface.
----
Inputs:
sim_time   - a datetime.timedelta object specifying the simulation time.
scenario   - a simglucose.scenario.Scenario object. Use
              simglucose.scenario_gen.RandomScenario or
              simglucose.scenario.CustomScenario to create a scenario object.
controller - a simglucose.controller.Controller object.
start_time - a datetime.datetime object specifying the simulation start time.
save_path  - a string representing the directory to save simulation results.
animate    - switch for animation. True/False.
parallel   - switch for parallel computing. True/False.
'''

if __name__ == '__main__':
    
    # strategy = 'PPO'
    strategy = 'BBC'
    # strategy = 'Random'
    # strategy = 'PID'
    
    # dizionario = {'paziente':['adult#001', 'adult#002', 'adult#003' , 'adult#004', 'adult#005',
    #                'adult#006', 'adult#007', 'adult#008' , 'adult#009', 'adult#010'],
    #              'ins_max':[0.08, 0.08, 0.08 , 0.06, 0.08,
    #                         0.09, 0.06, 0.07 , 0.07, 0.07]}
    
    # df_cap = pd.DataFrame(dizionario)
    # df_cap.to_excel(os.path.join(strategy_path,'paz_cap.xlsx'),index=False)
    # cap = df_cap.loc[df_cap['paziente']=='adult#007'].iloc[:,1]
    # cap = cap.iloc[0]
    # cap = float(df_cap.loc[df_cap['paziente']=='adult#010'].iloc[:,1])

    # n_days = 2
    n_days = 5
    n_hours = n_days*24
    seed = 42
    # ma = 15
    ma = 1
    training = 1440
    ripetizioni = 2
    # patient_names = ['adult#009']#,'adult#003']
    start_time = datetime.strptime('3/4/2022 12:00 AM', '%m/%d/%Y %I:%M %p')
    # AGGIUNGERE RIPETIZIONI (20 PER ORA) E LO SCENARIO FISSO
    # PER ORA TEST CON 5 GIORNI (POI PASSEREMO A 10)
    # CONTROLLARE CHE TD1SIMENV SIA ALLINEATO A PPOSIMENV
    # CONTROLLARE SE PRODUCE FILE
    
    
    cgm_name = 'Dexcom'
    insulin_pump_name = 'Nuovo'
    start_time = newdatetime
    animate = True
    parallel = True
    
    opt_dict = {
                # 'adult#001':[0.07],
                # 'adult#002':[0.07],
                # 'adult#003':[0.07],
                # 'adult#004':[0.07],
                # 'adult#005':[0.07],
                # 'adult#006':[0.14],
                'adult#007':[0.7],
                # 'adult#008':[0.07],
                # 'adult#009':[0.19],
                # 'adult#010':[0.05,0.06,0.08,0.09,0.10,0.11,0.12,0.13,0.14],
                }
    
    for p, cap in list(opt_dict.items()):
    
        for c in cap:
   
            print('training', p, c)
            
            dizionario = {'paziente': p,
                          'ins_max': c}
    
            df_cap = pd.DataFrame(dizionario, index=[0])
            df_cap.to_excel(os.path.join(strategy_path,'paz_cap.xlsx'),index=False)
            
            # for i, scen in zip(range(ripetizioni), scenarios.values()):
                   
            scen = [tuple(x) for x in list(scenarios.values())[7]]
            scenario = CustomScenario(start_time=start_time, scenario=scen)
 
            my_sim_time = timedelta(hours=float(n_hours))
            # scen_long = [(12, 100), (20, 120), (23, 30), (31, 40), (36, 70), (40, 100), (47, 10)] # scenario di due giorni
            # scen_long = create_scenario(n_days)
            # scenario = CustomScenario(start_time=start_time, scenario=scen_long)#, seed=seed)
            # scenario = RandomScenario(start_time=newdatetime)#, seed=seed)
            if strategy == 'PPO':
                # model_ppo = os.path.join(model_path, "ppo_sim_mod_food_hour_001_10000tmstp_insmax008")
                # model_ppo = os.path.join(model_path, 'ppo_sim_mod_food_hour_'+patient_names[0]+'_10000tmstp_1e-05_insmax009')
                model_ppo = os.path.join(model_path, 'ppo_sim_mod_food_hour_'+p+'_tmstp'+str(training)+'_lr00003_insmax'+str(c).replace('.','')+'_customscen')       
                
                controller = PPOController(model=PPO.load(model_ppo), target=140, window_size=ma)
                
            elif strategy == 'Random':
                controller = RandomController()
            elif strategy == 'BBC':
                # insulin_pump_name = 'Insulet'
                patient = T1DPatient.withName(p)
                sensor = CGMSensor.withName('Dexcom', seed=1)
                pump = InsulinPump.withName('Insulet')
                env = T1DSimEnv(patient=patient,
                                sensor=sensor,
                                pump = pump,
                                scenario=scenario)
                                # strategy=strategy)  
                controller = BBController()
                S1 = SimObj(env, controller, timedelta(days=n_days), animate=animate, path=data_path)
            elif strategy == 'PID':
                # insulin_pump_name = 'Insulet'
                patient = T1DPatient.withName(p)
                sensor = CGMSensor.withName('Dexcom', seed=1)
                pump = InsulinPump.withName('Insulet')
                env = T1DSimEnv(patient=patient,
                                sensor=sensor,
                                pump = pump,
                                scenario=scenario)
                                # strategy=strategy)
                controller = PIDController(P=0.001, I=0.00001, D=0.001, target=140)
                S1 = SimObj(env, controller, timedelta(days=n_days), animate=animate, path=data_path)
                parallel = False
                    
            
            df_strategy = pd.DataFrame({'strategy': strategy, 'patient': [p]})
        
            df_strategy.to_excel(os.path.join(strategy_path,'strategy.xlsx'),index=False)
            
            simulate(sim_time=my_sim_time,
                    scenario=scenario,
                    controller=controller,
                    patient_names=[p],
                    cgm_name=cgm_name,
                    cgm_seed=seed,
                    insulin_pump_name=insulin_pump_name,
                    start_time=start_time,
                    save_path=data_path,
                    animate=animate,
                    parallel=parallel)
                    # strategy=strategy)

