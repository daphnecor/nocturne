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
from torch.utils.data import Dataset

from tqdm import tqdm
import wandb

from bc_models import BehavioralCloningAgentJoint
import utils
from behavioral_sweep_config import BehavioralCloningSettings
from behavioral_sweep_config import scenario_config

wandb.login()
logging.basicConfig(level=logging.INFO)


def train(train_loader, model, optimizer, device, args):
    model.train()  # Set the model to training mode
    mean_loss = 0.0
    mean_acc = 0
    mean_dist_entropy = 0

    for states, expert_actions_idx in train_loader:
        # Get states and expert actions
        states, expert_actions_idx = states.to(device), expert_actions_idx.to(device)

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
    mean_loss /= len(train_loader)
    mean_acc /= len(train_loader)
    mean_dist_entropy /= len(train_loader)

    return mean_loss, mean_acc, mean_dist_entropy


def validate(val_loader, model, device, args):
    model.eval()  # Set the model to evaluation mode
    mean_loss = 0.0
    mean_acc = 0

    with torch.no_grad():
        for states, expert_actions_idx in val_loader:
            states, expert_actions_idx = states.to(device), expert_actions_idx.to(device)

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

        mean_loss /= len(val_loader)
        mean_acc /= len(val_loader)

    return mean_loss, mean_acc


class CustomDataset(Dataset):
    def __init__(self, observations_file, expert_actions_file):
        self.observations = torch.load(observations_file)
        self.expert_actions = torch.load(expert_actions_file)
        
        # Assuming both tensors have the same length
        assert len(self.expert_actions) == len(self.observations), "Expert actions and observations must have the same length."

    def __len__(self):
        return len(self.expert_actions)

    def __getitem__(self, index):
        observation = self.observations[index]
        expert_action = self.expert_actions[index].long()
        
        return observation, expert_action

def main():
    
    # Log basic settings
    basic_settings = {**scenario_config, **asdict(args)}    

    # Initialize
    run = wandb.init(config=basic_settings, magic=True)
    artifact = wandb.Artifact(name='bc_model_grid', type='model')

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
    
    train_dataset = CustomDataset(
        observations_file=EXPERT_OBS_PATH,
        expert_actions_file=EXPERT_ACTIONS_PATH,
    )
    
    val_dataset = CustomDataset(
        observations_file=EXPERT_OBS_PATH,
        expert_actions_file=EXPERT_ACTIONS_PATH,
    )

    # Create a DataLoader to load batches of data
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    obs, _ = next(iter(train_loader))
    state_dim = obs.shape[-1]

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
    
    # Training loop
    for epoch in range(EPOCHS):
        train_loss, train_acc, dist_entropy = train(train_loader, model, optimizer, device, args)
        val_loss, val_acc = validate(val_loader, model, device, args)

        # # # # # # End of epoch # # # # # #  
        wandb.log({
            "charts/epoch": epoch,
            "charts/act_dist_entropy": dist_entropy,
            "losses/train_loss": train_loss,
            "losses/train_accuracy": train_acc,
            "losses/val_loss": val_loss,
            "losses/val_accuracy": val_acc,
        })

        if epoch % 5 == 0:
            model_path = os.path.join(wandb.run.dir, f"BC_model.pt")
            torch.save(
                obj={
                    "model_state_dict": model.state_dict(),
                    "hidden_layers": HIDDEN_LAYERS,
                    "actions_discretizations": [args.accel_disc, args.steering_disc],
                    "actions_bounds": expert_bounds,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
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

    EXPERT_OBS_PATH = "/scratch/dc4971/nocturne/experiments/imitation_learning/data_expert_grid/observations.pt"
    EXPERT_ACTIONS_PATH = "/scratch/dc4971/nocturne/experiments/imitation_learning/data_expert_grid/expert_actions.pt"

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
            'epochs': {'values': [2000]},
            'batch_size': {'values': [128]},
            'hidden_layers': {'values': [[1024, 512, 256], [1024, 512, 128], [1024, 512, 448]]}, 
            'lr': { 'values': [1e-5, 5e-5, 1e-4, 5e-4]},
        },
    }
    
    # Initialize
    sweep_id = wandb.sweep(
        sweep=sweep_config, 
        project="behavioral_cloning_discrete",
    )

    # Run sweeps
    wandb.agent(sweep_id, function=main, count=5)
