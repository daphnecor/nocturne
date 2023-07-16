import logging
import json
import os
import pickle
import random
import time
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pdb
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from bc_models import BehavioralCloningAgentJoint
from waymo_data_loader import WaymoDataset
import utils
from behavioral_sweep_config import BehavioralCloningSettings
from behavioral_sweep_config import scenario_config

wandb.login()
logging.basicConfig(level=logging.INFO)


def train(train_loader, model, optimizer, batch_iters, device, args):
    model.train()  # Set the model to training mode
    mean_loss = 0.0
    mean_acc = 0
    mean_dist_entropy = 0

    for batch_idx in range(batch_iters):
        # Get states and expert actions
        states, expert_actions_idx = next(train_loader)
        states, expert_actions_idx = states.to(device), expert_actions_idx[0].to(device)

        # Zero param gradients
        optimizer.zero_grad()

        # Forward: Get selected actions log probs and the action distribution
        model_action_idx, action_dists = model(states)

        # Get log probs
        log_prob_batch = model.get_log_probs(
            expert_actions_idx
        )

        accuracy = (model_action_idx == expert_actions_idx).sum() / model_action_idx.shape[0]

        # Compute loss
        loss = -log_prob_batch.mean()    

        # Backward
        loss.backward()

        # Gradient clipping
        total_norm = utils.get_total_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)

        # Optimize
        optimizer.step()

        mean_loss += loss.item()
        mean_acc += accuracy.item()
        mean_dist_entropy += action_dists.entropy().mean().item()

    # End of epoch
    mean_loss /= batch_iters
    mean_acc /= batch_iters
    mean_dist_entropy /= batch_iters

    return mean_loss, mean_acc, mean_dist_entropy


def validate(val_loader, model, batch_iters, device, args):
    model.eval()  # Set the model to evaluation mode
    mean_loss = 0.0
    mean_acc = 0

    with torch.no_grad():
        for batch_idx in range(batch_iters):
            states, expert_actions_idx = next(val_loader)

            states, expert_actions_idx = states.to(device), expert_actions_idx[0].to(device)

            # Forward: Get selected actions log probs and the action distribution
            model_action_idx, action_dists = model(states)

            # Get log probs
            log_prob_batch = model.get_log_probs(
                expert_actions_idx
            )

            # Compute loss
            loss = -log_prob_batch.mean()

            # Compute accuracy
            accuracy = (model_action_idx == expert_actions_idx).sum() / model_action_idx.shape[0]

            mean_loss += loss.item()
            mean_acc += accuracy.item()

        mean_loss /= batch_iters
        mean_acc /= batch_iters

    return mean_loss, mean_acc


def main():
    
    # Log basic settings
    basic_settings = {**scenario_config, **asdict(args)}    

    # Initialize
    run = wandb.init(config=basic_settings, magic=True)
    artifact = wandb.Artifact(name='bc_model', type='model')

    # Sweep params
    BATCH_SIZE = wandb.config.batch_size
    HIDDEN_LAYERS = wandb.config.hidden_layers
    LR = wandb.config.lr
    EPOCHS = wandb.config.epochs

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
    
    logging.info(f'Action space: acc {[args.accel_lb, args.accel_ub]}, steering {[args.steering_lb, args.steering_ub]}')

    dataloader_config = {
        'tmin': args.tmin,
        'tmax': args.tmax,
        'accel_lb': args.accel_lb,
        'accel_ub': args.accel_ub,
        'accel_disc': args.accel_disc,
        'steering_lb': args.steering_lb,
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
        single_scene=args.single_scene,
    )

    train_loader = iter(
        DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            num_workers=args.n_cpus,
            pin_memory=True,
        ))
    
    val_dataset = WaymoDataset(
        data_path=args.valid_data_path,
        file_limit=args.num_files,
        dataloader_config=dataloader_config,
        scenario_config=scenario_config,
        single_scene=args.single_scene,
    )

    val_loader = iter(
        DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=args.n_cpus,
            pin_memory=True,
        ))

    sample_state, _ = next(train_loader)
    state_dim = sample_state.shape[-1]

    logging.info(f"Observation dim: {state_dim}")

    # Build model
    model = BehavioralCloningAgentJoint(
        num_states=state_dim, 
        hidden_layers=HIDDEN_LAYERS,
        actions_discretizations=[args.accel_disc, args.steering_disc],
        actions_bounds=expert_bounds,
        device=device, 
    ).to(device)

    # Create optimizer
    optimizer = Adam(model.parameters(), lr=LR)
    
    batch_iters = args.samples_per_epoch // BATCH_SIZE

    # Training loop
    for epoch in range(EPOCHS):
        train_loss, train_acc, dist_entropy = train(train_loader, model, optimizer, batch_iters, device, args)
        val_loss, val_acc = validate(val_loader, model, batch_iters, device, args)

        # # # # # # End of epoch # # # # # #  
        wandb.log({
            "charts/epoch": epoch,
            "charts/act_dist_entropy": dist_entropy,
            "losses/train_loss": train_loss,
            "losses/train_accuracy": train_acc,
            "losses/val_loss": val_loss,
            "losses/val_accuracy": val_acc,
        })

        if epoch % 2 == 0:
            model_path = os.path.join(wandb.run.dir, f"BC_model.pt")
            torch.save(
                obj={
                    "iter": iter,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                f=model_path,
            )
            logging.critical(f"\nSaved model at {model_path}")

    # Save trained Imitation Learning model as an artifact
    artifact.add_file(local_path=model_path)
    run.log_artifact(artifact)

if __name__ == '__main__':

    # Shared experiment settings
    args = BehavioralCloningSettings()

    sweep_config = {
        'method': 'random',
        'name': 'behavioral_cloning_sweep',
        'metric': {
            'goal': 'minimize',
            'name': 'loss'
        },
        'parameters': {
            'epochs': {'values': [10]},
            'batch_size': {'values': [512, 1024, 2048]},
            'hidden_layers': {'values': [[1024, 512, 256], [1024, 512, 128], [1024, 512, 448]]}, 
            'lr': { 'values': [1e-5, 5e-5, 1e-4, 5e-4]},
        },
    }
    
    # Initialize
    sweep_id = wandb.sweep(
        sweep=sweep_config, 
        project="behavioral_cloning",
        )

    # Run sweeps
    wandb.agent(sweep_id, function=main, count=1)
