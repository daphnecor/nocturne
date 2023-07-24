from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from collections import defaultdict

import waymo_data_loader 
from filters import MeanStdFilter
from torch.distributions.categorical import Categorical
from nocturne import Simulation

class BehavioralCloningAgentJoint(nn.Module):
    """Simple Behavioral Cloning class with a joint action space.

    Args:
        num_states (int): Number of environment states.
        hidden_layers (list of int): List containing the number of neurons for each hidden layer.
        actions_discretizations (list of int): List containing the number of discretizations for each action dimension.
        actions_bounds (list of tuple): List of tuples containing the minimum and maximum bounds for each action dimension.
        device (str): Device where the agent's model will be stored and trained (e.g., 'cpu' or 'cuda').
        deterministic (bool, optional): Flag indicating whether to choose actions deterministically (default: False).
    """

    def __init__(
        self, 
        num_states, 
        hidden_layers,
        actions_discretizations,
        actions_bounds,
        device, 
        deterministic=False,
    ):
        super(BehavioralCloningAgentJoint, self).__init__()
        self.num_states = num_states
        self.hidden_layers = hidden_layers
        self.actions_discretizations = actions_discretizations
        self.actions_bounds = actions_bounds
        self.device = device
        self.deterministic = deterministic
        
        # Create an action space
        self.action_grids = [
            torch.linspace(a_min, a_max, a_count, requires_grad=False).to(self.device)
            for (a_min, a_max), a_count in zip(self.actions_bounds, self.actions_discretizations)
        ]
        self._build_model()

    def _build_model(self):
        """Build agent MLP"""
         
        self.neural_net = nn.Sequential(
            MeanStdFilter(self.num_states),  # Pass states through filter
            nn.Linear(self.num_states, self.hidden_layers[0]),
            nn.Tanh(),
            *[
                nn.Sequential(
                    nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]),
                    nn.Tanh(),
                ) for i in range(len(self.hidden_layers) - 1)
            ],
        )
        
        # Map model representation to discrete action distributions
        pre_head_size = self.hidden_layers[-1]
        self.head = nn.Linear(pre_head_size, np.prod(self.actions_discretizations))

    def get_log_probs(self, expert_actions):
        """Return log probabilities of pi(expert_actions | state).

        Args:
            expert_actions (Tensor): Tensor representing the expert's actions.

        Returns:
            Tensor: Log probabilities of the expert's actions given the state.
        """
        return self.action_dist.log_prob(expert_actions)

    def forward(self, state):
        """Forward pass through the BC model.

        Args:
            state (Tensor): Input tensor representing the state of the environment.

        Returns:
            tuple: A tuple containing the selected action indices and the joint action distribution.
        """
            
        # Feed state to nn model
        outputs = self.neural_net(state)

        # Get joint action distribution
        action_logits = self.head(outputs)

        # Dist
        self.action_dist = Categorical(logits=action_logits)

        if self.deterministic:
            action_idx = action_logits.argmax(dim=-1)
        else:
            action_idx = self.action_dist.sample()

        # Return action distribution in state
        return action_idx, self.action_dist