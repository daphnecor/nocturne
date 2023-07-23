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


class PPOAgent(nn.Module):
    """Actor critic model class."""

    def __init__(self, state_dim, action_dim, hidden_layers, deterministic=False):
        super().__init__()
        self.deterministic = deterministic
        self.actor = self._build_model(state_dim, action_dim, hidden_layers, out_std=0.01)
        self.critic = self._build_model(state_dim, 1, hidden_layers, out_std=1.0)

    def _build_model(self, input_size, output_size, hidden_layers, out_std):
        """Build agent MLP"""

        layers = []
        for hl_idx, hl_size in enumerate(hidden_layers):
            layers.append(layer_init(nn.Linear(input_size, hl_size)))
            layers.append(nn.Tanh())
            input_size = hl_size
        # Output layer
        layers.append(layer_init(nn.Linear(hl_size, output_size), std=out_std))
        return nn.Sequential(*layers)

    def get_value(self, obs):
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None):
        logits = self.actor(obs)
        action_dist = Categorical(logits=logits)
        if action is None:
            action = action_dist.sample()
        return action, action_dist.log_prob(action), action_dist.entropy(), self.critic(obs)

    def get_policy(self, obs):
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        return probs

    def get_action(self, obs):
        logits = self.actor(obs)
        action_dist = Categorical(logits=logits)
        if self.deterministic:
            action_idx = action_dist.argmax(dim=-1)
        else:    
            action_idx = action_dist.sample()

        return action_idx, action_dist
