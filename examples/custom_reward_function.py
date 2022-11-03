import gym
from gym.envs.registration import register
import random

def custom_reward(BG_last_hour):
    if BG_last_hour[-1] > 180:
        return -1
    elif BG_last_hour[-1] < 70:
        return -2
    else:
        return 1


register(
    id='simglucose-adolescent2-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adolescent#002',
            'reward_fun': custom_reward}
)

env = gym.make('simglucose-adolescent2-v0')

reward = 1
done = False

observation = env.reset()
bolo_list = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
for t in range(200):
    env.render(mode='human')
    
    rand_idx = random.randrange(len(bolo_list))
    random_num = bolo_list[rand_idx]
    action = random_num
    
    # action = env.action_space.sample() # azione random tra low=0; high=30
    observation, reward, done, info = env.step(action)
    print(observation)
    print("Reward = {}".format(reward))
    if done:
        print("Episode finished after {} timesteps".format(t + 1))
        break
