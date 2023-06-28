import logging
import json
import os
import pickle
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from bc_model import BehavioralCloningAgentJoint
from waymo_data_loader import WaymoDataset
import utils
from behavioral_sweep_config import BehavioralCloningSettings
from behavioral_sweep_config import sweep_config, scenario_config

wandb.login()

logging.basicConfig(level=logging.INFO)

# Shared experiment settings
args = BehavioralCloningSettings()


def main():
  
    # Initialize
    run = wandb.init(magic=True)

    # Seed everything
    if args.seed is not None:
        utils.set_seed_everywhere(args.seed)

    # Get device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logging.critical(f"DEVICE: {device}")

    # Create dataset and dataloader
    expert_bounds = [
        [args.accel_lb, args.accel_ub], 
        [args.steering_lb, args.steering_ub],
    ]

    dataloader_config = {
        'tmin': args.tmin,
        'tmax': args.tmax,
        'view_dist': args.view_dist,
        'view_angle': args.view_angle,
        'dt': args.dt,
        'expert_action_bounds': expert_bounds,
        'expert_position': args.actions_are_positions,
        'state_normalization': args.state_normalization,
        'n_stacked_states': args.n_stacked_states,
    }

    dataset = WaymoDataset(
        data_path=args.train_path,
        file_limit=args.num_files,
        dataloader_config=dataloader_config,
        scenario_config=scenario_config,
    )

    data_loader = iter(
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.n_cpus,
            pin_memory=True,
        ))

    sample_state, _ = next(data_loader)
    state_dim = sample_state.shape[-1]

    # Build model
    model = BehavioralCloningAgentJoint(
        num_states=state_dim, 
        hidden_layers=wandb.config.hidden_layers,
        actions_discretizations=[args.accel_disc, args.steering_disc],
        actions_bounds=expert_bounds,
        device=device, 
    ).to(device)

    # Create optimizer
    optimizer = Adam(model.parameters(), lr=wandb.config.lr)

    # # Save configs
    # configs = {
    #     'scenario_config': scenario_cfg,
    #     'dataloader_config': dataloader_cfg,
    #     'model_config': model_cfg,
    # }

    batch_iters = args.samples_per_epoch // wandb.config.batch_size

    for epoch in range(args.epochs):
        for batch_idx in range(batch_iters):
            
            # Get states and expert actions
            states, expert_actions = next(data_loader)
            states, expert_actions = states.to(device), expert_actions.to(device)

            # Zero param gradients
            optimizer.zero_grad()

            # Forward: Get selected actions log probs and the action distribution
            action_idx, log_prob, action_dist = model(states)

            # Compute loss
            loss = -log_prob

            # Backward
            loss.backward()

            # Gradient clipping
            total_norm = utils.get_total_norm(model)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                args.clip_value,
            )
    
            # Optimize
            optimizer.step()
        
            # # # # # # Logging # # # # # # 
            wandb.log({
               # "charts/step": epoch,
                "losses/train_loss": loss.item(),        
            })


if __name__ == '__main__':
    
    # Initialize
    sweep_id = wandb.sweep(
        sweep=sweep_config, 
        project="behavioral_cloning",
        )

    # Run sweeps
    wandb.agent(sweep_id, function=main, count=5)

