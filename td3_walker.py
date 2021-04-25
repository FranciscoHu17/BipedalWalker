import gym
import numpy as np
import torch 

from statistics import mean, median
from collections import Counter


def simulate(env, num_iterations, learning_rate, goal_steps, score_req):  
  training = []
  scores = []
  accepted = []
  for game in range(num_iterations):
    score = 0
    curr_run = []
    observation = env.reset()

    for t in range(goal_steps):
        env.render()
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

  print("Accepted Mean:", mean(accepted))
  print("Accepted Median:", median(accepted))
  print("All Mean:", mean(scores))
  print("All Median:", median(scores))


def main():
    print('Running TD3 algorithm on BipedalWalker-v3.')
    env = gym.make('BipedalWalker-v3')

    LR = 1e-3   # Learning Rate
    goal_steps = 500
    score_req = -100
    iterations = 10

    simulate(env, iterations, LR, goal_steps, score_req)


main()
