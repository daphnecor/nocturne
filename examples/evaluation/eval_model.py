import sys
import os

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
import yaml
import hydra

from examples.imitation_learning.bc_model import BehavioralCloningAgent
import utils
from loader import SimulationDataset
from loader import simulation_collate_fn

# Read the YAML file
with open('cfgs/config.yaml', 'r') as yaml_file:
    args = yaml.safe_load(yaml_file)

@hydra.main(config_path="../../cfgs/imitation", config_name="config")
def main(args):
    #BASE_PATH = '/scratch/dc4971/nocturne/data'
    #valid_data_path = Path(f'{BASE_PATH}/formatted_json_v2_no_tl_valid')
    valid_data_paths = list(Path(args.data.test_path).glob('tfrecord*.json'))[:args.num_eval_files]

    scenario_cfg = {
        'start_time': args.scene.start_time,
        'allow_non_vehicles': args.scene.allow_non_vehicles,
        'spawn_invalid_objects': args.scene.spawn_invalid_objects,
        'max_visible_road_points': args.scene.max_visible_road_points,
        'sample_every_n': args.scene.sample_every_n,
        'road_edge_first': args.scene.road_edge_first,
    }

    scene_dataset = SimulationDataset(
        paths = valid_data_paths,   
        scenario_config=scenario_cfg,
    )

    scene_loader = DataLoader(
        batch_size=len(valid_data_paths),
        dataset=scene_dataset,
        num_workers=0,
        collate_fn=simulation_collate_fn
    )
    
    # Get state space dimension
    state_dim = 35110

    # Load model
    # Note: doing it this way for now because torch.load() expects model.py 
    # to be in the same folder
    device = 'cpu'
    model = torch.load(args.best_bc_model_path).to(device)
    model.eval()

    collision_rate_veh = np.zeros(len(valid_data_paths))
    collision_rate_road_edge = np.zeros(len(valid_data_paths))
    goal_rates = np.zeros(len(valid_data_paths))

    for batch in scene_loader:
        # Process the batch of simulation objects
        for idx, sim_obj in tqdm(enumerate(batch)):
            print(f' Evaluating model on traffic scene: {idx}')
            # Access and work with individual simulation objects
            # Perform operations or computations on 'sim'
            collision_rate_veh[idx], collision_rate_road_edge[idx], goal_rates[idx] = utils.evaluate_agent_in_traffic_scene_(
            sim_obj, scenario_cfg, num_stacked_states=5, model=model,
            )

    print(f'Collision rates: \n (veh <> veh) {collision_rate_veh.mean() * 100:.2f} % \n (veh <> road objects) {collision_rate_road_edge.mean() * 100:.2f} % \n')
    print(f'Goal rates: \n Vehicles that reach their goal {goal_rates.mean() * 100:.2f} %')

if __name__ == '__main__':
    main()
