import numpy as np
import gym
import torch
import random
from td3 import TD3
from buffer import ExperienceReplay
import matplotlib.pyplot as plt

def main():
    env = gym.make('BipedalWalker-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    buffer_size = 1000000
    batch_size = 100
    noise = 0.1

    policy = TD3(state_dim, action_dim, max_action, env)
    try:
        policy.load()
    except Exception as e:
        print('No previous model to load.')

    buffer = ExperienceReplay(buffer_size, batch_size)

    episodes = 5000
    timesteps = 2000

    curr_reward = []

    for episode in range(episodes):
        avg_reward = 0
        state = env.reset()
        for i in range(timesteps):
            # Same as the TD3, select an action and add noise:
            if(i% 100 == 0):
                print('Finished', i, 'timesteps')
            action = policy.select_action(state) + np.random.normal(0, max_action * noise, size=action_dim)
            action = action.clip(env.action_space.low, env.action_space.high)
            # Make an action. 
            next_state, reward, done, _ = env.step(action)
            buffer.store_transition(state, action, reward, next_state, done)
            state = next_state
            avg_reward += reward
            env.render()
            if(len(buffer) > batch_size):
                policy.train(buffer, i)
            if(done or i > timesteps):
                print('Episode ', episode,'finished with reward:', avg_reward)
                print('Finished at timestep ', i)
                curr_reward.append(avg_reward)
                break
        print('Finished 2000 timesteps')
        if(np.mean(curr_reward[-100:]) >= 300):
            policy.save()
            break

        if(episode % 100 == 0 and episode > 0):
            #Save policy and optimizer every 100 episodes
            policy.save()
    

    fig = plt.figure()
    plt.plot(np.arange(1, len(ep_reward) + 1), ep_reward)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    



main()