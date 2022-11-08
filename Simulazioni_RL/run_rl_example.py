from simglucose.simulation.user_interface import simulate
from simglucose.simulation.scenario_gen import RandomScenario
# from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.random_ctrller import RandomController
from datetime import datetime
from datetime import timedelta

now = datetime.now() # gestire una qualsiasi data di input
newdatetime = now.replace(hour=12, minute=00)

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
    seed = 42
    my_sim_time = timedelta(hours=float(4))
    scenario = RandomScenario(start_time=newdatetime, seed=seed)
    controller = RandomController()
    patient_names = ['adult#001']#,'adult#002']
    cgm_name = 'Dexcom'   
    insulin_pump_name = 'Nuovo'
    start_time = newdatetime
    save_path = 'Risultati'
    animate = True
    parallel = True

    simulate(sim_time=my_sim_time,
            scenario=scenario,
            controller=controller,
            patient_names=patient_names,
            cgm_name=cgm_name,
            cgm_seed=seed,
            insulin_pump_name=insulin_pump_name,
            start_time=start_time,
            save_path=save_path,
            animate=animate,
            parallel=parallel)
