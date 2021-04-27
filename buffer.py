import numpy as np
import torch
import random
from collections import deque


class ExperienceReplay:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size= batch_size

    def __len__(self):
        return len(self.buffer)

    # Add a transition to the memory by basic SARNS convention. 
    def store_transition(self, state, action, reward, new_state, done):
        # If buffer is abuot to overflow, begin rewriting existing memory? 
        self.buffer.append((state, action, reward, new_state, done))

    # Sample only the memory that has been stored. Samples BATCH
    # amount of samples. 
    def sample(self):
        sample = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*sample)
        states = torch.from_numpy(np.array(states, dtype=np.float32))
        actions = torch.from_numpy(np.array(actions, dtype=np.float32))
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32).reshape(-1, 1))
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32))
        dones = torch.from_numpy(np.array(dones, dtype=np.uint8).reshape(-1, 1)).float()
        return (states, actions, rewards, next_states, dones)