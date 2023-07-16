import os
from dataclasses import dataclass
from pathlib import Path

DATA_FOLDER = '/scratch/dc4971/nocturne/data'

@dataclass
class BehavioralCloningSettings:
    # fmt: off
    seed: int = None # 12
    accel_disc: int = 5
    accel_lb: int = -2
    accel_ub: int = 2
    steering_lb: int = -0.7                   # Corresponds to about 40 degrees of max steering angle
    steering_ub: int = 0.7                    # Corresponds to about 40 degrees of max steering angle
    steering_disc: int = 5
    joint_act_space: bool = True
    num_files: int = -1 
    tmin: int = 0
    tmax: int = 90
    n_stacked_states: int = 1 # Agent memory
    view_dist: int = 80
    view_angle: int = 3.14
    dt: float = 0.1
    state_normalization: int = 100
    lr: float = 3e-4                                  # Learning rate
    samples_per_epoch: int = 50_000                   # Number of batch iterations per epoch
    batch_size: int = 512
    epochs: int = 10
    discrete: bool = True # Actions are discrete
    clip_value: int = 1
    actions_are_positions: bool = False
    n_cpus: int = 0
    train_data_path = Path(f'{DATA_FOLDER}/formatted_json_v2_no_tl_train')
    valid_data_path = Path(f'{DATA_FOLDER}/formatted_json_v2_no_tl_valid')
    valid_data_paths = list(Path(valid_data_path).glob('tfrecord*.json'))
    single_scene: str = "example_scenario.json"

scenario_config = {
    'start_time': 0,
    'allow_non_vehicles': False,
    'spawn_invalid_objects': True,
    'max_visible_road_points': 500,
    'sample_every_n': 1,
    'road_edge_first': False,
}