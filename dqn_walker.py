import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from collections import deque
import random
import pickle
import datetime


class ExperienceReplay:
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)

    def __len__(self):
        return len(self.memory)

    # Add a transition to the memory
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    # Sample only the memory that has been stored
    def sample_memory(self, sample_size):
        sample = random.sample(self.memory, sample_size)
        states, actions, rewards, next_states, dones = zip(*sample)

        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

def build_dqn(lr, n_actions, input_shape, fc1_shape, fc2_shape):
    model = keras.Sequential([
        #keras.layers.Dense(input_shape[0], activation='relu'),
        keras.layers.Dense(fc1_shape, activation='relu'),
        keras.layers.Dense(fc2_shape, activation='relu'),
        keras.layers.Dense(n_actions, activation='tanh')
    ])

    model.compile(optimizer=Adam(learning_rate=lr), loss ='mean_squared_error')

    return model

class Agent:
    def __init__(self, lr, gamma, action_space, epsilon, sample_size, input_shape,
                 epsilon_dec=1e-3, epsilon_end=0.01, mem_size=1000000, file_name='dqn_model'):
                self.action_space = action_space
                self.gamma = gamma
                self.epsilon = epsilon
                self.eps_dec = epsilon_dec
                self.eps_min = epsilon_end
                self.sample_size = sample_size
                self.model_file = file_name
                self.memory = ExperienceReplay(mem_size)
                self.dqn_model = build_dqn(lr, action_space.shape[0], input_shape, 400, 300)

    def step(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        
        # Start learning only when there are enough sample sizes
        if len(self.memory) > self.sample_size:
            self.learn()

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = self.action_space.sample()
        else:
            state = np.array([observation])
            #print(state.shape)
            #print(observation)
            actions = self.dqn_model.predict(state)
            #print(actions[0])
            #action = np.argmax(actions[0])
            action = actions[0]

        return action
    
    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample_memory(self.sample_size)

        q_eval = self.dqn_model.predict(states)
        q_next = self.dqn_model.predict(next_states)    
        #q_target = np.copy(q_eval)

        sample_index = np.arange(self.sample_size, dtype=np.int32)
        q_target = rewards + (self.gamma*np.max(q_next, axis=1))*dones

        self.dqn_model.train_on_batch(states, q_target)
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_model(self):
        self.dqn_model.save(self.model_file)

    def load_model(self):
        self.dqn_model = load_model(self.model_file)


def replay_actions(env, actions):
    done = False
    state = env.reset()
    for action in actions:
        env.render()
        env.step(action)
    env.close()

def store_actions(model, actions):
    actions_file = open(model+'_actions','wb')
    pickle.dump(actions, actions_file)
    actions_file.close()

def load_actions(model):
    actions_file = open(model+'_actions','rb')
    actions = pickle.load(actions_file)
    actions_file.close()

    return actions


if '__main__' == __name__:
    tf.compat.v1.disable_eager_execution()
    env = gym.make('BipedalWalker-v3')
    lr = 1e-4
    n_games = 10000

    agent = Agent(lr=lr, gamma=0.99, action_space=env.action_space, epsilon=1.0,
                 sample_size=64, input_shape=env.observation_space.shape)
    
    load = input("load from model? [y/n]: ")
    if load == "y":
        load = input("input file name to load: ")
        agent.model_file = load
        agent.load_model()
        actions = load_actions(load)
        print()
        print("loading model from", load)
        print()
        replay_actions(env, actions)
    
    max_score = -10000
    max_game = 0
    scores = []
    eps_history = []

    start = datetime.datetime.now()

    for game in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        game_actions = [] # actions taken during this game
        episode_start = datetime.datetime.now()

        while not done:
            #if (game+1)%50 == 0: 
            #    env.render()
            action = agent.choose_action(observation)
            game_actions.append(action)
            next_observation, reward, done, info = env.step(action)
            agent.step(observation, action, reward, next_observation, done)
            
            score += reward
            observation = next_observation

        eps_history.append(agent.epsilon)
        scores.append(score)
        episode_end = datetime.datetime.now()

        avg_score = np.mean(scores[-100:])
        
        if score > max_score:
            max_score = score
            max_game = game
            replay_actions(env, game_actions)

            agent.model_file = "dqn_walker"
            agent.save_model()
            store_actions(agent.model_file, game_actions)
            print("saving model as", agent.model_file)
            print()

        elapsed = episode_end - episode_start

        print('game:', game)
        print('reward:', str(score))
        print("max game:", str(max_game), "\tmax reward:", str(max_score))
        print('average score for the last 100 games:', avg_score)
        print('time:', str(elapsed.total_seconds()),'seconds')
        print()

    end = datetime.datetime.now()
    elapsed = end - start

    print('Total time:',elapsed.total_seconds(), 'seconds')