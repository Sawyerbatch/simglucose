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

env = T1DSimGymnasiumEnv_MARL(
    patient_name='adult#001',
    custom_scenario=scenario,
    reward_fun=new_reward,
    seed=123,
    render_mode="human",
)

# Crea un modello PPO per ogni agente
ppo_models = {agent: PPO("MlpPolicy", env, verbose=1) for agent in env.agents}

# Definisci il numero di passi da eseguire nella simulazione
num_steps = 5000

observation, info = env.reset(seed=42)

# Esegui la simulazione
for step in range(num_steps):
    print('Step numero', step)
    actions = {}
    for agent in env.agents:
        action, _ = ppo_models[agent].predict(observation[agent], deterministic=True)
        actions[agent] = action

    print(actions)
    observations, rewards, done, truncations, infos = env.step(actions)
    env.render()
    time.sleep(0.1)

    if any(done.values()):
        print(f"La simulazione è terminata dopo {step+1} passi")
        break
