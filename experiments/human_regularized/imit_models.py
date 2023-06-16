import numpy as np
import torch
import torch.nn as nn

from examples.imitation_learning.filters import MeanStdFilter
from torch.distributions.categorical import Categorical
from nocturne import Simulation


class BehavioralCloningAgentJoint(nn.Module):
    """Simple Behavioral Cloning class."""

    def __init__(self, num_inputs, config, device):
        super(BehavioralCloningAgentJoint, self).__init__()
        self.num_states = num_inputs
        self.hidden_layers = config.hidden_layers
        self.actions_discretizations = config.actions_discretizations
        self.actions_bounds = config.actions_bounds
        # Create an action space
        self.action_grids = [
            torch.linspace(a_min, a_max, a_count, requires_grad=False).to(device)
            for (a_min, a_max), a_count in zip(
                self.actions_bounds, self.actions_discretizations
            )
        ]
        self._build_model()

    def _build_model(self):
        """Build agent MLP"""

        # Create neural network model
        self.neural_net = nn.Sequential(
            MeanStdFilter(self.num_states),  # Pass states through filter
            nn.Linear(self.num_states, self.hidden_layers[0]),
            nn.Tanh(),
            *[
                nn.Sequential(
                    nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]),
                    nn.Tanh(),
                )
                for i in range(len(self.hidden_layers) - 1)
            ],
        )

        # Map model representation to discrete action distributions
        pre_head_size = self.hidden_layers[-1]
        self.head = nn.Linear(pre_head_size, np.prod(self.actions_discretizations))

    def forward(self, state, deterministic=False):
        """Forward pass through the BC model.

        Args:
            state (Tensor): Input tensor representing the state of the environment.
            deterministic (bool): Flag indicating whether to choose actions deterministically.

        Returns:
            Tensor: A tensor representing the joint action distribution.
        """

        # print(f'input: {state.shape}')

        # Feed state to nn model
        outputs = self.neural_net(state)

        # print(f'after nn: {outputs.shape}')

        # Get joint action distribution
        action_logits = self.head(outputs)

        # print(f'action_logits: {action_logits.shape}')

        # Dist
        action_dist = Categorical(logits=action_logits)

        if deterministic:
            action_idx = action_logits.argmax(dim=-1)
        else:
            action_idx = action_dist.sample()

        # Return action, log_prob, and full distribution
        return action_idx, action_dist.log_prob(action_idx), action_dist