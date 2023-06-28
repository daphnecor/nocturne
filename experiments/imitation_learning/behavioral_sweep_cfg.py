import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BehavioralCloningConstants:
    # fmt: off
    seed: int = 12
    accel_discretization: int = 15
    accel_lb: int = -3
    accel_ub: int = 2
    steering_lb: int = -0.7                   # Corresponds to about 40 degrees of max steering angle
    steering_ub: int = 0.7                    # Corresponds to about 40 degrees of max steering angle
    steering_discretization: int = 15 
   



# Define 
sweep_config = {
    'method': 'random',
    'name': 'behavioral_cloning_sweep',
    'metric': {
        'goal': 'minimize',
        'name': 'train_loss'
    },
    'parameters': {
        'batch_size': {'values': [1, 2, 3]},
    }
}

