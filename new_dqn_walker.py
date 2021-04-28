import numpy as np
import torch
import random
import gym
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime


ENV = "BipedalWalker-v3"
MODEL_FILE = "dqn_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
N_GAMES = 5000
MEM_SIZE = 1000000
BATCH_SIZE = 64
TARGET_UPDATE = 10
GAMMA = 0.99
EPSILON = 1
EPSILON_DEC = 1e-3
EPSILON_MIN = 0.01
LR = 1e-4






class ExperienceReplay:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    # Add a transition to the memory by basic SARNS convention. 
    def store_transition(self, state, action, reward, new_state, done):
        # If buffer is abuot to overflow, begin rewriting existing memory? 
        self.buffer.append((state, action, reward, new_state, done))

    # Sample only the memory that has been stored. Samples BATCH
    # amount of samples. 
    def sample(self):
        sample = random.sample(self.buffer, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*sample)
        states = torch.tensor(states).float().to(DEVICE)
        actions = torch.stack(actions).long().to(DEVICE)
        rewards = torch.tensor(rewards).float().to(DEVICE)
        next_states = torch.tensor(next_states).float().to(DEVICE)
        dones = torch.tensor(dones).float().to(DEVICE)
        return (states, actions, rewards, next_states, dones)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(QNetwork, self).__init__()
        # Make a simple 3 later linear network
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = self.l1(state)
        x = F.relu(x)
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

class Agent():
    # Initialize the agent
    def __init__(self, state_space, action_space):
        self.memory = ExperienceReplay(MEM_SIZE)
        self.action_space = action_space
        self.main_model = QNetwork(state_space.shape[0], action_space.shape[0], action_space.high[0]).to(DEVICE)
        self.target_model = QNetwork(state_space.shape[0], action_space.shape[0], action_space.high[0]).to(DEVICE)
        self.optimizer = optim.Adam(self.main_model.parameters(), lr=LR)

        # Target model will be a copy of the main model and will not be trained
        self.target_model.load_state_dict(self.main_model.state_dict())
        self.target_model.eval()


    # Agent saves its experiences and learn
    def step(self, state, action, reward, new_state, done):
        # Stores the transition into Experience Replay
        self.memory.store_transition(state, action, reward, new_state, done)

        # Agent will only learn when there are enough experiences
        if len(self.memory) > BATCH_SIZE:
            self.learn()

    # Agent learns
    def learn(self):
        # Sample random minibatch of transitions from Experience Replay
        state, action, reward, new_state, done = self.memory.sample()

        # Computes Q(s_{curr},a') then chooses columns of actions that were takenfor each batch
        q_eval = self.main_model(state).gather(1, action)
        
        # Clone the model and use it to generate Q learning targets for the main model
        # Also predicts the max Q value for the next state
        q_next = self.target_model(new_state).max(1)[0].detach()

        # Q learning targets = r if next state is terminal or
        # Q learning targets = r + GAMMA*(Q(s_{next},a')) if next state is not terminal
        q_target = reward + GAMMA*(q_next) *(1-done)
        
        print(self.main_model(state).size(), self.target_model(new_state).size())
        exit(0)

        # Compute MSE loss
        loss = F.mse_loss(q_eval, q_target.unsqueeze(1))

        # Gradient descent on the loss function and does backpropragation
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.main_model.parameters():
            # Clip the error term to be between -1 and 1
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()



    # Action chosen is either a random action or based on the Bellman Equation
    def choose_action(self, state):
        # With probability EPSILON, select a random action
        if np.random.random() < EPSILON:
            return torch.from_numpy(self.action_space.sample())
        # Otherwise select the action with the highest Q value
        else: 
            state = torch.FloatTensor(state.reshape(1,-1)).to(DEVICE)
            
            # action that maximizes r + GAMMA*(Q*(s',a')) based on optimal Q*(s',a')  
            with torch.no_grad():
                return self.main_model(state).max(1)[1].view(1,1)

    
    






def main():
    env = gym.make(ENV)
    state_space = env.observation_space
    action_space = env.action_space
    agent = Agent(state_space, action_space)

    max_score = -10000
    max_game = 0
    scores = []
    start = datetime.datetime.now()

    for game in range(N_GAMES):
        done = False
        score = 0
        observation = env.reset()
        game_actions = [] # actions taken during this game
        episode_start = datetime.datetime.now()

        while not done:
            # Depending on probability of EPSILON, either select a random action or select an action based on the Bellman Equation
            action = agent.choose_action(observation)
            
            # Execute the action in env and observe reward and next state
            next_observation, reward, done, info = env.step(action)

            # Stores experiences and learns
            agent.step(observation, action, reward, next_observation, done)

            # Update the score and state of this run
            score += reward
            observation = next_observation
          
        # Every TARGET_UPDATE games, reset the target model to the main model
        if game % TARGET_UPDATE == 0:
            agent.target_model.load_state_dict(agent.main_model.state_dict())

        episode_end = datetime.datetime.now()
        elapsed = episode_end - episode_start
        scores.append(score)
        avg_score = np.mean(scores[-100:])

        if score > max_score:
            max_score = score
            max_game = game

        print('game:', game)
        print('reward:', str(float(score)))
        print("max game:", str(max_game), "\tmax reward:", str(float(max_score)))
        print('average score for the last 100 games:', avg_score)
        print('time:', str(elapsed.total_seconds()),'seconds')
        print()

        



        


main()