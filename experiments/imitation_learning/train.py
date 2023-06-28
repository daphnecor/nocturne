# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Imitation learning training script (behavioral cloning)."""
from datetime import datetime
from pathlib import Path
import pickle
import random
import json
import os
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from waymo_data_loader import WaymoDataset
from bc_model import BehavioralCloningAgent
import utils

logging.basicConfig(level=logging.INFO)

from behavioral_sweep_cfg import BehavioralCloningConstants, sweep_config
import wandb
wandb.login()

args = BehavioralCloningConstants()


def train():
    """A behavioral cloning run."""

    # Initialize
    run = wandb.init(
        magic=True,
    )

    # Seed everything
    if args.seed is not None:
        utils.set_seed_everywhere(args.seed)

    # Get device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logging.critical(f"DEVICE: {device}")

    # Create dataset and dataloader
    expert_bounds = [[-6, 6], [-0.7, 0.7]]
    actions_bounds = expert_bounds
    action_bounds = [[args.accel_lb, args.accel_ub], [args.steering_lb, args.steering_ub]]
    #actions_discretizations = [15, 43]
    mean_scalings = [3, 0.7]
    std_devs = [0.1, 0.02]



    dataloader_cfg = {
        'tmin': args.data.tmin,
        'tmax': args.data.tmax,
        'view_dist': args.data.view_dist,
        'view_angle': args.data.view_angle,
        'dt': args.data.dt,
        'expert_action_bounds': expert_bounds,
        'expert_position': args.actions_are_positions,
        'state_normalization': args.data.state_normalization,
        'n_stacked_states': args.data.n_stacked_states,
    }

    scenario_cfg = {
        'start_time': args.scene.start_time,
        'allow_non_vehicles': args.scene.allow_non_vehicles,
        'spawn_invalid_objects': args.scene.spawn_invalid_objects,
        'max_visible_road_points': args.scene.max_visible_road_points,
        'sample_every_n': args.scene.sample_every_n,
        'road_edge_first': args.scene.road_edge_first,
    }
    
    # # # # # # CREATE DATA LOADER # # # # # #
    dataset = WaymoDataset(
        data_path=args.data.train_path,
        file_limit=args.data.num_files,
        dataloader_config=dataloader_cfg,
        scenario_config=scenario_cfg,
    )

    data_loader = iter(
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.n_cpus,
            pin_memory=True,
        ))

    # # # # # # BUILD MODEL # # # # # #
    sample_state, _ = next(data_loader)
    state_dim = sample_state.shape[-1]

    model_cfg = {
        'n_inputs': state_dim,
        'hidden_layers': [1024, 256, 128],
        'discrete': args.discrete,
        'mean_scalings': mean_scalings,
        'std_devs': std_devs,
        'actions_discretizations': actions_discretizations,
        'actions_bounds': actions_bounds,
    }

    model = BehavioralCloningAgent(
        num_inputs=state_dim,
        config=model_cfg,
        device=device,
    ).to(device)
    model.train()

    for i in range(4):
        wandb.log({'t' : i})

    # # Create optimizer
    # optimizer = Adam(model.parameters(), lr=args.lr)

    # # Create export directory
    # time_str = datetime.now().strftime('%Y_%m_%d')
    # exp_dir = Path.cwd() / Path('train_logs') / time_str
    # exp_dir.mkdir(parents=True, exist_ok=True)

    # # Save configs
    # configs_path = exp_dir / 'configs.json'
    # configs = {
    #     'scenario_cfg': scenario_cfg,
    #     'dataloader_cfg': dataloader_cfg,
    #     'model_cfg': model_cfg,
    # }
    # with open(configs_path, 'w') as fp:
    #     json.dump(configs, fp, sort_keys=True, indent=4)
    # print('Wrote configs at', configs_path)

    # # Wandb logging
    # if args.wandb:
    #     wandb_mode = args.wandb_mode
    #     wandb.init(
    #         config=args,
    #         project=args.wandb_project,
    #         #name=args.experiment,
    #         group=args.wandb_group,
    #         resume=args.wandb_resume,
    #         settings=wandb.Settings(start_method="fork"),
    #         mode=wandb_mode
    #     )

    # print('Exp dir created at', exp_dir)

    # start = time.time()
    
    # # # # # # # TRAIN LOOP # # # # # #
    # for epoch in range(args.epochs):
    #     print(f'\nepoch {epoch+1}/{args.epochs}')
    #     n_samples = epoch * args.batch_size * (args.samples_per_epoch //
    #                                            args.batch_size)

    #     for i in tqdm(range(args.samples_per_epoch // args.batch_size), unit='batch'):
            
    #         # Get states and expert actions
    #         states, expert_actions = next(data_loader)
    #         states, expert_actions = states.to(device), expert_actions.to(device)

    #         # Forward
    #         # Get taken actions by model, their indices and distributions over actions
    #         model_actions, model_action_idx, action_dists_in_state = model(states)

    #         # Compute log probabilities and indices of the expert actions
    #         log_prob, expert_action_idx = utils.compute_log_prob(
    #             action_dists=action_dists_in_state,
    #             ground_truth_action=expert_actions,
    #             action_grids=model.action_grids,
    #             reduction='mean',
    #             return_indexes=True,
    #         )

    #         # Compute loss
    #         loss = -log_prob.sum()

    #         metrics_dict = {}

    #         # Optimize 
    #         optimizer.zero_grad()
    #         loss.backward()

    #         # Gradient clipping
    #         total_norm = utils.get_total_norm(model)
    #         torch.nn.utils.clip_grad_norm_(
    #             model.parameters(), 
    #             args.clip_value,
    #         )
    
    #         # Optimize
    #         optimizer.step()

    #         # # # # # # LOGGING # # # # # #
    #         metrics_dict['train/grad_norm'] = total_norm
    #         metrics_dict['train/loss'] = loss.item()
    #         metrics_dict['train/accel_logprob'] = log_prob[0]
    #         metrics_dict['train/steer_logprob'] = log_prob[1]
            
    #         accuracy = [
    #             (model_idx == expert_idx).float().mean(axis=0)
    #             for model_idx, expert_idx in zip(model_action_idx, expert_action_idx.T)
    #         ]

    #         metrics_dict['train/accel_acc'] = accuracy[0]
    #         metrics_dict['train/steer_acc'] = accuracy[1]

    #         # WANDB
    #         if args.wandb:
    #             wandb.log(metrics_dict)

    #     # # # # # # # # # # # # # # # # # # # # # # # #

    #     # Save model checkpoint
    #     if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
    #         model_path = exp_dir / f'model_{epoch+1}_epochs.pth'
    #         # Save model
    #         #torch.save(model.state_dict(), str(model_path))
    #         torch.save(model, str(model_path))
    #         pickle.dump(filter, open(exp_dir / f"filter_{epoch+1}.pth", "wb"))
    #         print(f'\nSaved model at {model_path}')

    # # # # # # END EPOCHS # # # # # 
    # print('Done, exp dir is', exp_dir)
    # print(f'Total training time: {time.time() - start:.3f} seconds')

if __name__ == '__main__':
    
    sweep_id = wandb.sweep(
        sweep=sweep_config, 
        project="behavioral_cloning",
        )

    wandb.agent(sweep_id, function=train, count=3)

