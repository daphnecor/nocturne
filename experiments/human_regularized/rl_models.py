"""
PPO Policy and value networks. 
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    """Actor critic model class."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, 1024)),
            nn.Tanh(),
            layer_init(nn.Linear(1024, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        ) 

        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, 1024)),
            nn.Tanh(),
            layer_init(nn.Linear(1024, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_policy(self, x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return probs
