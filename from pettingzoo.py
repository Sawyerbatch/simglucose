import pygame
from pettingzoo.butterfly import knights_archers_zombies_v10
from stable_baselines3 import PPO

env = knights_archers_zombies_v10.env(render_mode="human")
env.reset(seed=42)

manual_policy = knights_archers_zombies_v10.ManualPolicy(env)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    elif agent == manual_policy.agent:
        # get user input (controls are WASD and space)
        action = manual_policy(observation, agent)
    else:
        # this is where you would insert your policy (for non-player agents)
        action = env.action_space(agent).sample()

    env.step(action)
env.close()


# Crea l'ambiente parallelo
env = pistonball_v6.parallel_env()

# Utilizza SuperSuit per trasformare l'ambiente Petting Zoo in un ambiente vettorizzato
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')

# Crea il modello PPO
model = PPO("MlpPolicy", env, verbose=3)

# Addestra il modello
model.learn(total_timesteps=10)