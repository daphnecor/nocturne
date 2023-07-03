# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Dataloader for imitation learning in Nocturne."""
from collections import defaultdict
import random
from torch.utils.data import DataLoader
import utils
import torch
from pathlib import Path
import numpy as np

from cfgs.config import ERR_VAL
from nocturne import Simulation
from behavioral_sweep_config import BehavioralCloningSettings, scenario_config
import logging

logging.basicConfig(level=logging.INFO)

def _get_waymo_iterator(paths, data_config, scenario_config):
    
    # If worker has no paths, return an empty iterator
    if len(paths) == 0:
        return
    
    if data_config['joint_act_space']:
        # Create action grids 
        accel_grid = np.linspace(
                data_config['accel_lb'], data_config['accel_ub'], data_config['accel_disc']
        )
        steer_grid = np.linspace(
                data_config['steering_lb'], data_config['steering_ub'], data_config['steering_disc']
        )

        # Create joint action space
        actions_to_joint_idx = {}
        i = 0
        for accel in accel_grid:
            for steer in steer_grid:
                actions_to_joint_idx[accel, steer] = [i]
                i += 1   
    
    # Load dataloader config
    view_dist = data_config.get('view_dist', 80)
    view_angle = data_config.get('view_angle', np.radians(120))
    dt = data_config.get('dt', 0.1)
    expert_action_bounds = data_config['expert_action_bounds']
    expert_position = False
    state_normalization = data_config.get('state_normalization', 100)
    n_stacked_states = data_config.get('n_stacked_states', 5)

    while True:
        # select a random scenario path
        scenario_path = np.random.choice(paths)

        # create simulation
        sim = Simulation(str(scenario_path), scenario_config)
        scenario = sim.getScenario()

        # set objects to be expert-controlled
        for obj in scenario.getObjects():
            obj.expert_control = True

        # we are interested in imitating vehicles that moved
        objects_that_moved = scenario.getObjectsThatMoved()
        objects_of_interest = [
            obj for obj in scenario.getVehicles() if obj in objects_that_moved
        ]

        # initialize values if stacking states
        stacked_state = defaultdict(lambda: None)
        initial_warmup = n_stacked_states - 1

        state_list = []
        action_list = []

        # iterate over timesteps and objects of interest
        for time in range(data_config['tmin'], data_config['tmax']):
            for obj in objects_of_interest:
                # get state
                ego_state = scenario.ego_state(obj)
                visible_state = scenario.flattened_visible_state(
                    obj, view_dist=view_dist, view_angle=view_angle)
                state = np.concatenate((ego_state, visible_state))

                # normalize state
                state /= state_normalization

                # stack state
                if n_stacked_states > 1:
                    if stacked_state[obj.getID()] is None:
                        stacked_state[obj.getID()] = np.zeros(
                            len(state) * n_stacked_states, dtype=state.dtype)
                    stacked_state[obj.getID()] = np.roll(
                        stacked_state[obj.getID()], len(state))
                    stacked_state[obj.getID()][:len(state)] = state

                if np.isclose(obj.position.x, ERR_VAL):
                    continue

                if not expert_position: # Taking the acc and steering wheel angle!
                    # get expert action
                    expert_action = scenario.expert_action(obj, time)
                    # check for invalid action (because no value available for taking derivative)
                    # or because the vehicle is at an invalid state
                    if expert_action is None:
                        continue
                    expert_action = expert_action.numpy()
                    
                    acceleration = expert_action[0]
                    steering = expert_action[1]

                    # throw out actions containing NaN or out-of-bound values
                    if np.isnan(expert_action).any() \
                            or expert_action[0] < expert_action_bounds[0][0] \
                            or expert_action[0] > expert_action_bounds[0][1] \
                            or expert_action[1] < expert_action_bounds[1][0] \
                            or expert_action[1] > expert_action_bounds[1][1]:
                        continue

                else:
                    expert_pos_shift = scenario.expert_pos_shift(obj, time)
                    if expert_pos_shift is None:
                        continue
                    expert_pos_shift = expert_pos_shift.numpy()
                    expert_heading_shift = scenario.expert_heading_shift(
                        obj, time)
                    
                    if expert_heading_shift is None \
                            or expert_pos_shift[0] < expert_action_bounds[0][0] \
                            or expert_pos_shift[0] > expert_action_bounds[0][1] \
                            or expert_pos_shift[1] < expert_action_bounds[1][0] \
                            or expert_pos_shift[1] > expert_action_bounds[1][1] \
                            or expert_heading_shift < expert_action_bounds[2][0] \
                            or expert_heading_shift > expert_action_bounds[2][1]:
                        continue
                    
                    expert_action = np.concatenate(
                        (expert_pos_shift, [expert_heading_shift]))
                    
            
                # Convert to joint action
                if data_config['joint_act_space']:

                    # Map actions to nearest grid indices
                    accel_grid_idx = utils.find_closest_index(accel_grid, acceleration)
                    steering_grid_idx = utils.find_closest_index(steer_grid, steering)
                    
                    # Map to joint action space
                    expert_action = actions_to_joint_idx[accel_grid[accel_grid_idx], steer_grid[steering_grid_idx]]

                # yield state and expert action
                if stacked_state[obj.getID()] is not None:
                    if initial_warmup <= 0:  # warmup to wait for stacked state to be filled up
                        state_list.append(stacked_state[obj.getID()])
                        action_list.append(expert_action)
                else:
                    state_list.append(state)
                    action_list.append(expert_action)

            # step the simulation
            sim.step(dt)
            if initial_warmup > 0:
                initial_warmup -= 1

        if len(state_list) > 0:
            temp = list(zip(state_list, action_list))
            random.shuffle(temp)
            state_list, action_list = zip(*temp)
            for state_return, action_return in zip(state_list, action_list):
                yield (state_return, action_return)


class WaymoDataset(torch.utils.data.IterableDataset):
    """Waymo dataset loader."""

    def __init__(self,
                 data_path,
                 dataloader_config={},
                 scenario_config={},
                 file_limit=None):
        super(WaymoDataset).__init__()

        # save configs
        self.dataloader_config = dataloader_config
        self.scenario_config = scenario_config

        # get paths of dataset files (up to file_limit paths)
        self.file_paths = list(
            Path(data_path).glob('tfrecord*.json'))[:file_limit]
        print(f'WaymoDataset: loading {len(self.file_paths)} files.')

        # sort the paths for reproducibility if testing on a small set of files
        self.file_paths.sort()

    def __iter__(self):
        """Partition files for each worker and return an (state, expert_action) iterable."""
        # get info on current worker process
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # single-process data loading, return the whole set of files
            return _get_waymo_iterator(self.file_paths, self.dataloader_config,
                                       self.scenario_config)

        # distribute a unique set of file paths to each worker process
        worker_file_paths = np.array_split(
            self.file_paths, worker_info.num_workers)[worker_info.id]
        return _get_waymo_iterator(list(worker_file_paths),
                                   self.dataloader_config,
                                   self.scenario_config)


if __name__ == '__main__':

    args = BehavioralCloningSettings()

    expert_bounds = [
        [args.accel_lb, args.accel_ub], 
        [args.steering_lb, args.steering_ub],
    ]

    dataloader_config = {
            'tmin': args.tmin,
            'tmax': args.tmax,
            'accel_lb': args.accel_lb,
            'accel_ub': args.accel_ub,
            'accel_disc': args.accel_disc,
            'steering_lb': args.steering_disc,
            'steering_ub': args.steering_ub,
            'steering_disc': args.steering_disc,
            'view_dist': args.view_dist,
            'view_angle': args.view_angle,
            'dt': args.dt,
            'expert_action_bounds': expert_bounds,
            'joint_act_space': args.joint_act_space,
            'expert_position': args.actions_are_positions,
            'state_normalization': args.state_normalization,
            'n_stacked_states': args.n_stacked_states,
    }
    
    train_dataset = WaymoDataset(
        data_path=args.train_data_path,
        file_limit=args.num_files,
        dataloader_config=dataloader_config,
        scenario_config=scenario_config,
    )

    train_loader = iter(
        DataLoader(
            train_dataset,
            batch_size=5,
            num_workers=args.n_cpus,
            pin_memory=True,
        ))
    
    states, expert_actions = next(train_loader)

    print('hi')
    # for i, x in zip(range(100), data_loader):
    #     print(i, x[0].shape, x[1].shape)
