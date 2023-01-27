from simglucose.simulation.user_interface import simulate
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.random_ctrller import RandomController
from simglucose.controller.pid_ctrller import PIDController
from stable_baselines3 import PPO
from simglucose.controller.ppo_ctrller import PPOController
from datetime import datetime
from datetime import timedelta
from simglucose.simulation.env import T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
import os

now = datetime.now() # gestire una qualsiasi data di input
newdatetime = now.replace(hour=12, minute=00)

data = str(datetime.now()).replace(" ", "_" ).replace("-", "" ).replace(":", "" )[:8]

os.chdir('C:\GitHub\simglucose\Simulazioni_RL\Risultati')
cwd = os.getcwd()

data_path = os.path.join(cwd, data)  
if not os.path.exists(data_path):
    os.makedirs(data_path)

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
    # model_ppo = "ppo_sim_mod_food_hour_10000tmstp_buono"
    model_ppo = "ppo_sim_mod_food_hour_10000tmstp_moltobuono"
    # seed = 42
    my_sim_time = timedelta(hours=float(240))
    scenario = RandomScenario(start_time=newdatetime)#, seed=seed)
    # controller = RandomController()
    
    # controller = PIDController(P=0.001, I=0.00001, D=0.001, target=140)
    controller = PPOController(model=PPO.load(model_ppo))
    patient_names = ['adult#002']
    # patient_names = ['adult#00'+str(i) for i in range(2,10)] + \
    #     ['adult#010']
    # patient = T1DPatient.withName('adult#001')
    # sensor = CGMSensor.withName('Dexcom', seed=1)
    # pump = InsulinPump.withName('Nuovo')
    cgm_name = 'Dexcom'   
    insulin_pump_name = 'Nuovo'
    start_time = newdatetime
    save_path = data_path
    animate = True
    parallel = True
    # env = T1DSimEnv(patient=patient,
    #                 sensor=sensor,
    #                 pump = pump,
    #                 scenario=scenario)   
    
    # controller = BBController()
    
    simulate(sim_time=my_sim_time,
            scenario=scenario,
            controller=controller,
            patient_names=patient_names,
            cgm_name=cgm_name,
            # cgm_seed=seed,
            insulin_pump_name=insulin_pump_name,
            start_time=start_time,
            save_path=save_path,
            animate=animate,
            parallel=parallel)
    
    # s1 = SimObj(env, controller, timedelta(days=1),
    #             animate=False, path=save_path)
    # results = sim(s1)
    # print(results)