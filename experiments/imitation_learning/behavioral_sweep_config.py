import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BehavioralCloningSettings:
    # fmt: off
    seed: int = 12
    accel_disc: int = 15
    accel_lb: int = -3
    accel_ub: int = 2
    steering_lb: int = -0.7                   # Corresponds to about 40 degrees of max steering angle
    steering_ub: int = 0.7                    # Corresponds to about 40 degrees of max steering angle
    steering_disc: int = 15 
    num_files: int = -1 
    tmin: int = 0
    tmax: int = 90
    n_stacked_states: int = 5 # Agent memory
    view_dist: int = 80
    view_angle: int = 3.14
    dt: float = 0.1
    state_normalization: int = 100
    lr: float = 3e-4                                    # Learning rate
    samples_per_epoch: int = 50_000                   # Number of batch iterations per epoch
    batch_size: int = 512
    epochs: int = 100
    discrete: bool = True # Actions are discrete
    clip_value: int = 1
    actions_are_positions: bool = False
    n_cpus: int = 0
    train_path: str = '/scratch/dc4971/nocturne/data/formatted_json_v2_no_tl_train'
    test_path: str = '/scratch/dc4971/nocturne/data/formatted_json_v2_no_tl_valid'


scenario_config = {
    'start_time': 0,
    'allow_non_vehicles': False,
    'spawn_invalid_objects': True,
    'max_visible_road_points': 500,
    'sample_every_n': 1,
    'road_edge_first': False,
}

sweep_config = {
    'method': 'random',
    'name': 'behavioral_cloning_sweep',
    'metric': {
        'goal': 'minimize',
        'name': 'train_loss'
    },
    'parameters': {
        'batch_size': {'values': [512, 1024, 2048]},
        'hidden_layers': {'values': [[1024, 256, 128], [1024, 512, 128]]},
        'lr': {'max': 1e-3, 'min': 1e-5},
    },
}
