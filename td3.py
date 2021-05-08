import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from architecture import Actor, Critic
from buffer import ExperienceReplay

# Config
ENV = "BipedalWalker-v3"
BATCH_SIZE = 100
DISCOUNT_FACTOR = 0.99
EXPLORE_POLICY = 0.1
LEARN_RATE = .001
POLICY_DELAY = 2
TAU = 0.005
NOISE_POLICY = 0.2
NOISE_CLIP = 0.5

class TD3():
    def __init__(self, state_dim, action_dim, max_action, env, device):
        super(TD3, self).__init__()

        # Set up Actor net
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LEARN_RATE)
        self.device = device

        # Set up Critic net
        self.critic = Critic(state_dim, action_dim).to(device) # only needs state + action
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LEARN_RATE)
        self.max_action = max_action
        self.env = env

    def select_action(self, state, noise=0.1): 
        # Gets best action to take based on current state/policy.
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        # Add a random amount of noise from a normal dist to the action. 
        if(noise == EXPLORE_POLICY): 
            action = (action + np.random.normal(0, noise, size=self.env.action_space.shape[0]))

        return self.actor(state).cpu().data.numpy().flatten()


    # Change path and filename of where you'd like to save the model to.
    # Default saves to 'td3_actor_finalized.pth'
    def save(self):
        torch.save(self.actor.state_dict(), './td3_actor_finalized.pth')
        torch.save(self.critic.state_dict(), './td3_actor_finalized.pth')
        return
    
    # Default loads a 600-episode trained agent to resume training/display on. 
    def load(self):
        self.actor.load_state_dict(torch.load('./pretrained/td3_actor300ep.pth',  map_location=torch.device('cpu')))
        self.critic.load_state_dict(torch.load('./pretrained/td3_critic300ep.pth',  map_location=torch.device('cpu')))
        return

    def train(self, replay_buffer, current_iteration): 
        # Pseudocode detailed by :
        # http://bicmr.pku.edu.cn/~wenzw/bigdata/lect-dyna3w.pdf

        # Randomly sample batch (n = 100) of transitions from replay replay_buffer D.
        # All SARNS + done are already Tensors. 
        state, action, reward, next_state, done = replay_buffer.sample()
        # Find the target action.
        # noise = sampled from N(0, sigma), where sigma = NOISE_POLICY (0.2).
        # Clips all values to be in the range of noise_clip (-0.5, 0.5).
        # https://stackoverflow.com/questions/44417227/make-values-in-tensor-fit-in-given-range
        tensor_cpy = action.clone().detach()
        noise = tensor_cpy.normal_(0, NOISE_POLICY).clamp(-NOISE_CLIP, NOISE_CLIP)
        # noise = (torch.randn_like(action) * NOISE_POLICY).clamp(-NOISE_CLIP, NOISE_CLIP)

        # Clips the next action + clipped_noise with -max_action, +max_action.
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
        # Compute target Qs:
        # Runs forward pass with the next_state and the next_action, returns (Q1, Q2).
        # Softmax? Of Q1, Q2 (min i=1,2 of Q_target_i(s',a'(s')))
        target_q1, target_q2 = self.critic_target(next_state, next_action)
        target_q = ((torch.min(target_q1, target_q2)) * (1-done)) + reward
        curr_q1, curr_q2 = self.critic(state, action)

        # ... and then both are learned by regressing (MSE) to the target:
        critic_loss = F.mse_loss(curr_q1, target_q) + F.mse_loss(curr_q2, target_q)
        self.critic_optimizer.zero_grad() # reset any previously held grads to 0, else it accumulates
        critic_loss.backward()
        self.critic_optimizer.step() # Updates Q-functions by one gradient step.

        # The policy is learned by maximizing the Q every other iteration
        if (current_iteration % POLICY_DELAY == 0):
            # Update policy by one step of grad ascent. 
            # 1/|batch_size| sum(-self.critic(state, self.actor(state))[0])
            # self.critic(...)[0] gets Q1 calculation. 
            actor_loss = -self.critic(state, self.actor(state))[0].mean()

            # Update target networks: 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # If i % policy_delay == 0, then we update model (delayed updates)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):    
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

