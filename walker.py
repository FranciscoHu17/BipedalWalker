import gym
import numpy as np
import tensorflow as tf
from statistics import mean, median
from collections import Counter

def simulate():
    training = []
    scores = []
    accepted = []
    for game in range(games):
        score = 0
        curr_run = []
        observation = env.reset()
        
        for t in range(goal_steps):
            #env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            curr_run.append([observation, action])
            score += reward
            
            if done:
                # Agent fell to the ground
                print("Episode finished after {} timesteps".format(t+1))
                break
        
        scores.append(score)
        if score >= score_req:
            accepted.append(score)
            for data in curr_run:
                training.append(data)
        
    env.close()

    
    #neural_network(np.array(training), np.array(accepted))

    #print(Counter(accepted))
    print("Accepted Mean:", mean(accepted))
    print("Accepted Median:", median(accepted))
    print("All Mean:", mean(scores))
    print("All Median:", median(scores))


def neural_network(data, accepted):
    dataset = tf.data.Dataset.from_tensor_slices((data, accepted))

def train_model(data):
    X = 1

if '__main__' == __name__:
    env = gym.make('BipedalWalker-v3')

    LR = 1e-3   # Learning Rate
    goal_steps = 500
    score_req = -100
    games = 10
    
    simulate()

