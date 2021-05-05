import numpy as np
import sys
import gym
import torch
import random
from td3 import TD3
from buffer import ExperienceReplay
import matplotlib.pyplot as plt

def main():
    env = gym.make('BipedalWalker-v3')

    # set seed for reproducable results
    seed = 1
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    buffer_size = 1000000
    batch_size = 100
    noise = 0.1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    policy = TD3(state_dim, action_dim, max_action, env, device)
    # try:
    #     policy.load()
    # except Exception as e:
    #     print('No previous model to load.')

    buffer = ExperienceReplay(buffer_size, batch_size, device)

    episodes = 650
    timesteps = 2000

    best_reward = -1*sys.maxsize
    scores_over_episodes = []

    for episode in range(episodes):
        avg_reward = 0
        state = env.reset()
        for i in range(timesteps):
            # Same as the TD3, select an action and add noise:
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
                scores_over_episodes.append(avg_reward)
                print('Episode ', episode,'finished with reward:', avg_reward)
                print('Finished at timestep ', i)
                break
        if(np.mean(scores_over_episodes[-50:]) > 350):
            print('Saving agent- past 50 scores gave better average score than +350.')
            best_reward = np.mean(scores_over_episodes[-50:])
            policy.save()
            break # Saved agent so far. Break out of episodes and end. 

        if(episode >= 0 and avg_reward > best_reward):
            print('Saving agent- score for this episode was better than best-known score..')
            best_reward = avg_reward
            policy.save() # Save current policy + optimizer
    
    fig = plt.figure()
    plt.plot(np.arange(1, len(scores_over_episodes) + 1), scores_over_episodes)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


main()