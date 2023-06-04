from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from collections import defaultdict

from examples.imitation_learning import waymo_data_loader 
from examples.imitation_learning.filters import MeanStdFilter
from torch.distributions.categorical import Categorical
from nocturne import Simulation

class BehavioralCloningAgent(nn.Module):
    """Simple Behavioral Cloning class."""
    def __init__(self, num_inputs, config, device):
        super(BehavioralCloningAgent, self).__init__()
        self.num_states = num_inputs
        self.hidden_layers = config['hidden_layers']
        self.actions_discretizations = config['actions_discretizations']
        self.actions_bounds = config['actions_bounds']
        
        # Create an action space
        self.action_grids = [
            torch.linspace(a_min, a_max, a_count, requires_grad=False).to(device)
                for (a_min, a_max), a_count in zip(
                    self.actions_bounds, self.actions_discretizations)
        ]
        self._build_model()

    def _build_model(self):
        """Build agent MLP"""
        
        # Create neural network model
        self.neural_net = nn.Sequential(
            MeanStdFilter(self.num_states), # Pass states through filter
            nn.Linear(self.num_states, self.hidden_layers[0]),
            nn.Tanh(),
            *[
                nn.Sequential(
                    nn.Linear(self.hidden_layers[i],
                                self.hidden_layers[i + 1]),
                    nn.Tanh(),
                ) for i in range(len(self.hidden_layers) - 1)
            ],
        )
        
        # Map model representation to discrete action distributions
        pre_head_size = self.hidden_layers[-1]
        self.heads = nn.ModuleList([
            nn.Linear(pre_head_size, discretization)
            for discretization in self.actions_discretizations
        ])

    def forward(self, state, deterministic=False):
        """Forward pass through the BC model.

            Args:
                state (Tensor): Input tensor representing the state of the environment.

            Returns:
                Tuple[List[Tensor], List[Tensor], List[Categorical]]: A tuple of three lists:
                1. A list of tensors representing the actions to take in response to the input state.
                2. A list of tensors representing the indices of the actions in their corresponding action grids.
                3. A list of Categorical distributions over the actions.
            """

        # Feed state to nn model
        outputs = self.neural_net(state)

        # Get distribution over every action in action types (acc, steering, head tilt)
        action_dists_in_state = [Categorical(logits=head(outputs)) for head in self.heads]

        # Get action indices (here deterministic)
        # Find indexes in actions grids whose values are the closest to the ground truth actions
        actions_idx = [dist.logits.argmax(axis=-1) if deterministic else dist.sample()
                       for dist in action_dists_in_state]
        
        # Get action in action grids
        actions = [
            action_grid[action_idx] for action_grid, action_idx in zip(
                self.action_grids, actions_idx)
        ]
        
        return actions, actions_idx, action_dists_in_state