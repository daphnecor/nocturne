""" 
Training script for Human-Regularized PPO. 
"""
import datetime
import glob
import logging
import os
import random
import time
from typing import Any, Dict
from pathlib import Path

import yaml
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import wandb
from pprint import pprint
import pdb

from dataclasses import asdict, dataclass

import nocturne_gym as gym
from nocturne_gym import NocturneEnv
from constants import PPOExperimentConfig, WandBSettings, ScenarioConfig
from rl_models import PPOAgent
from bc_models import BehavioralCloningAgentJoint
import utils as ppo_utils

# Set wandb waiting time to avoid cluster timeout errors
os.environ["WANDB__SERVICE_WAIT"] = "300"
logging.basicConfig(level=logging.INFO)

def main():

    # Default configurations to be stored as metadata
    args = PPOExperimentConfig()
    args_wandb = WandBSettings()
    meta_data_dict = {**RL_ENV_ARGS, **asdict(args)}

    # Initialize run
    run = wandb.init()

    # Get sweep params
    NUM_ROLLOUTS = wandb.config.num_rollouts
    TOTAL_ITERS = wandb.config.total_iters
    LR = wandb.config.learning_rate
    ENT_COEF = wandb.config.ent_coef
    VF_COEF = wandb.config.vf_coef
    LAMBDA = wandb.config.lambda_hr
    HIDDEN_LAYERS = wandb.config.hidden_layers

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Make environment from selected traffic scene
    env = gym.NocturneEnv(
        path_to_scene=BASE_PATH,
        scene_name=TRAFFIC_SCENE,
        valid_veh_dict=valid_veh_dict,
        scenario_config=SCENARIO_CONFIG,
        cfg=RL_ENV_ARGS,
    )

    # Initialize actor and critic models
    obs_space_dim = env.observation_space.shape[0]
    act_space_dim = env.action_space.n

    # Build actor critic network
    ppo_agent = PPOAgent(
        obs_space_dim,
        act_space_dim,
        hidden_layers=HIDDEN_LAYERS,
    ).to(DEVICE)

    # Load human anchor policy
    bc_model_artifact = run.use_artifact(ARTIFACT_PATH, type="model")
    bc_model_artifact_dir = bc_model_artifact.download()
    BC_MODEL_CHECKPOINT = torch.load(
        f"{bc_model_artifact_dir}/BC_model.pt", map_location=DEVICE
    )

    human_anchor_policy = BehavioralCloningAgentJoint(
        num_states=obs_space_dim,
        hidden_layers=BC_MODEL_CHECKPOINT["hidden_layers"],
        actions_discretizations=BC_MODEL_CHECKPOINT["actions_discretizations"],
        actions_bounds=BC_MODEL_CHECKPOINT["actions_bounds"],
        device=DEVICE,
        deterministic=True,
    ).to(DEVICE)
    human_anchor_policy.load_state_dict(BC_MODEL_CHECKPOINT["model_state_dict"])
    human_anchor_policy.eval()
        
    # Optimizer 
    optimizer = optim.Adam(ppo_agent.parameters(), lr=LR, eps=1e-5)

    # Create descriptive model name
    MODEL_NAME = f"ppo_model_Dstate_{obs_space_dim}_Dact_{act_space_dim}_S{SCENE_NAME}"
    model_path = os.path.join(wandb.run.dir, f"{MODEL_NAME}.pt")
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    global_step = 0

    # Logging
    logging.info(f"DEVICE: {DEVICE}")
    logging.info(f"Using: {SCENE_NAME} with action_space_dim: {act_space_dim} and obs_space_dim: {obs_space_dim} \n")
    pprint(env.idx_to_actions)

    for iter_ in range(1, TOTAL_ITERS):

        start_iter = time.time()
        
        # Anneal learning rate 
        if args.anneal_lr:
            frac = 1.0 - (iter_ - 1.0) / TOTAL_ITERS
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Storage
        obs_tensor = torch.zeros((NUM_ROLLOUTS, args.num_steps, MAX_AGENTS, obs_space_dim)).to(DEVICE)
        rew_tensor = torch.zeros((NUM_ROLLOUTS, args.num_steps, MAX_AGENTS)).to(DEVICE)
        act_tensor = torch.zeros_like(rew_tensor).to(DEVICE)
        done_tensor = torch.zeros_like(rew_tensor).to(DEVICE)
        logprob_tensor = torch.zeros_like(rew_tensor).to(DEVICE)
        value_tensor = torch.zeros_like(rew_tensor).to(DEVICE)
        veh_coll_tensor = torch.zeros((NUM_ROLLOUTS)).to(DEVICE)
        edge_coll_tensor = torch.zeros((NUM_ROLLOUTS)).to(DEVICE)
        goal_tensor = torch.zeros((NUM_ROLLOUTS)).to(DEVICE)

        # (1) ROLLOUTS
        start_rollouts = time.time()
        for rollout_step in range(NUM_ROLLOUTS):

            buffer, veh_veh_collisions, veh_edge_collisions, veh_goal_achieved, num_agents = ppo_utils.do_policy_rollout(
                args, 
                env,
                ppo_agent,
                DEVICE,
            )

            global_step += 1

            # Store rollout scene experience
            obs_tensor[rollout_step, :, :num_agents, :] = ppo_utils.dict_to_tensor(
                buffer.observations
            ).to(DEVICE)
            rew_tensor[rollout_step, :, :num_agents] = ppo_utils.dict_to_tensor(
                buffer.rewards
            ).to(DEVICE)
            act_tensor[rollout_step, :, :num_agents] = ppo_utils.dict_to_tensor(
                buffer.actions
            ).to(DEVICE)
            done_tensor[rollout_step, :, :num_agents] = ppo_utils.dict_to_tensor(buffer.dones).to(
                DEVICE
            ).to(DEVICE)
            logprob_tensor[rollout_step, :, :num_agents] = ppo_utils.dict_to_tensor(
                buffer.logprobs
            ).to(DEVICE)
            value_tensor[rollout_step, :, :num_agents] = ppo_utils.dict_to_tensor(
                buffer.values
            ).to(DEVICE)

            # Normalize counts by agents in scene
            veh_coll_tensor[rollout_step] = veh_veh_collisions / num_agents
            edge_coll_tensor[rollout_step] = veh_edge_collisions / num_agents
            goal_tensor[rollout_step] = veh_goal_achieved / num_agents

            # Logging
            if args_wandb.track:
                wandb.log(
                    {
                        "charts/iter": iter_,
                        "charts/global_step": global_step,
                        "charts/num_agents_in_scene": num_agents,
                        "charts/close_agent(3)_episodic_return": sum(buffer.rewards[3]).item(),
                        "charts/far_agent(32)_episodic_return": sum(buffer.rewards[32]).item(),
                        "charts/total_episodic_return": sum(sum(buffer.rewards.values())).item(),
                        "charts/goal_achieved_rate": goal_tensor[rollout_step],
                        "charts/veh_veh_collision_rate": veh_coll_tensor[rollout_step],
                        "charts/veh_edge_collision_rate": edge_coll_tensor[rollout_step],
                    }
                )
                
        # # # # # # END ROLLOUTS # # # # # #
        time_rollouts = time.time() - start_rollouts

        # # # # Compute advantage estimate via GAE on collected experience # # # #
        # Flatten along the num_agents x policy rollout steps axes
        dones = done_tensor.permute(0, 2, 1).reshape(
            num_agents * NUM_ROLLOUTS, args.num_steps
        ).to(DEVICE)
        values = value_tensor.permute(0, 2, 1).reshape(
            num_agents * NUM_ROLLOUTS, args.num_steps
        ).to(DEVICE)
        rewards = rew_tensor.permute(0, 2, 1).reshape(
            num_agents * NUM_ROLLOUTS, args.num_steps
        ).to(DEVICE)
        logprobs = logprob_tensor.permute(0, 2, 1).reshape(
            num_agents * NUM_ROLLOUTS, args.num_steps
        ).to(DEVICE)
        actions = act_tensor.permute(0, 2, 1).reshape(
            num_agents * NUM_ROLLOUTS, args.num_steps
        ).to(DEVICE)

        obs_flat = obs_tensor.permute(0, 2, 1, 3).reshape(
            num_agents * NUM_ROLLOUTS,
            args.num_steps,
            obs_space_dim,
        ).to(DEVICE)
        dones_flat = done_tensor.permute(0, 2, 1).reshape(
            num_agents * NUM_ROLLOUTS, args.num_steps
        ).to(DEVICE)

        # Select the most recent observation for every policy rollout and agent
        last_step_alive = ppo_utils.find_last_zero_index(dones)
        next_obs = torch.stack(
            [
                obs_flat[idx, last_step_alive[idx], :]
                for idx in range(len(last_step_alive))
            ]
        ).to(DEVICE)

        # Assumption: all agents are done after 80 steps, even if they haven't
        # reached their target
        next_done = torch.ones(size=(num_agents * NUM_ROLLOUTS,)).to(DEVICE)

        # (2) GENERAL ADVANTAGE ESTIMATION
        with torch.no_grad():
            next_value = ppo_agent.get_value(next_obs).reshape(-1).to(DEVICE)
            advantages = torch.zeros(
                (MAX_AGENTS * NUM_ROLLOUTS, args.num_steps)
            ).to(DEVICE)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[:, t + 1]
                    nextvalues = values[:, t + 1]

                # Compute TD-error
                delta = (
                    rewards[:, t]
                    + args.gamma * nextvalues * nextnonterminal
                    - values[:, t]
                )
                # Update advantage for timestep
                lastgaelam = (
                    delta
                    + args.gamma
                    * args.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )
                advantages[:, t] = lastgaelam

            returns = advantages + values

        # Filter out invalid indices
        valid_samples_idx = torch.nonzero(dones_flat != 1, as_tuple=False)
        batch_size = len(valid_samples_idx)

        b_obs = obs_flat[valid_samples_idx[:, 0], valid_samples_idx[:, 1], :]
        b_logprobs = logprobs[valid_samples_idx[:, 0], valid_samples_idx[:, 1]]
        b_actions = actions[valid_samples_idx[:, 0], valid_samples_idx[:, 1]]
        b_advantages = advantages[valid_samples_idx[:, 0], valid_samples_idx[:, 1]]
        b_returns = returns[valid_samples_idx[:, 0], valid_samples_idx[:, 1]]
        b_values = values[valid_samples_idx[:, 0], valid_samples_idx[:, 1]]

        b_inds = np.arange(batch_size)
        clipfracs = []
        minibatch_size = batch_size // num_agents
        
        # (3) OPTIMIZATION
        start_optim = time.time()
        for epoch in range(args.update_epochs):

            # Shuffle batch indices
            np.random.shuffle(b_inds)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # Compute kl divergence to the human anchor policy
                action_idx, tau_dist = human_anchor_policy(b_obs[mb_inds])
                actor_dist = ppo_agent.get_policy(b_obs[mb_inds])

                kl_div_to_human_policy = kl_loss(tau_dist.probs.log(), actor_dist.probs)
                # Use below to verify correctness, pytorch implementation is faster
                #kl_div_scipy = scipy_kl_batch(actor_dist, tau_dist)

                # Check if the minibatch has at least two elements
                if len(mb_inds) < 2: continue  # Skip this minibatch 

                # Compute new logprobs of state dist
                _, newlogprob, entropy, newvalue = ppo_agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )

                # Compute r(theta) between old and new policy
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()


                # Compute final loss
                LOSS_IL = LAMBDA * (kl_div_to_human_policy)
                LOSS_PPO = (1 - LAMBDA) * (pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss)

                loss = LOSS_IL + LOSS_PPO

                # Backwards 
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ppo_agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # Logging: end of iteration
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        explained_var = np.nan if np.var(y_true) == 0 else 1 - (np.var(y_true - y_pred)) / np.var(y_true)

        if args_wandb.track:
            wandb.log(
                {
                    "charts/batch_size": batch_size,
                    "charts/b_advantages": wandb.Histogram(b_advantages.cpu().numpy()),
                    "charts/b_advantages_mean": b_advantages.mean(),
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "charts/LOSS_PPO": LOSS_PPO,
                    "charts/LOSS_IL": LOSS_IL,
                    "losses/kl_div_to_human_policy": kl_div_to_human_policy.item(),
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/kl_policy_update": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                }
            )

        # Profiling
        time_optim = time.time() - start_optim
        time_iter = time.time() - start_iter
        if args_wandb.track:
            wandb.log(
                {
                    "iter": iter_,
                    "profiling/rollout_step_frac": time_rollouts / time_iter,
                    "profiling/optim_step_frac": time_optim / time_iter,
                }
            )

        # Save model BC_MODEL_CHECKPOINT in wandb directory
        if iter_ % SAVE_MODEL_FREQ == 0:

            # Create model artifact
            model_artifact = wandb.Artifact(
                name=f"{MODEL_TYPE}_iter_{iter_}", 
                type=MODEL_TYPE,
                description=f"HR-PPO on scene: {SCENE_NAME}",
                metadata=dict(meta_data_dict),
            )
            
            # Save
            torch.save(
                obj={
                    "iter": iter_,
                    "model_state_dict": ppo_agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "obs_space_dim": obs_space_dim,
                    "act_space_dim": act_space_dim,
                    "hidden_layers": HIDDEN_LAYERS,
                    "policy_loss": pg_loss,
                    "lambda_hr": LAMBDA,
                    "kl_div_to_human_policy": kl_div_to_human_policy,
                    "minibatch_size": minibatch_size,
                },
                f=model_path,
            )

            # Save model artifact  
            model_artifact.add_file(local_path=model_path)
            wandb.save(model_path, base_path=wandb.run.dir)
            run.log_artifact(model_artifact)

            logging.info(f"Stored {MODEL_TYPE} after {iter_} iters.")

    # End of run
    env.close()


if __name__ == "__main__":

    # Set DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path to traffic scenario + name
    BASE_PATH = "/scratch/dc4971/nocturne/data/formatted_json_v2_no_tl_train/"
    TRAFFIC_SCENE = "example_scenario.json"
    SCENARIO_CONFIG = asdict(ScenarioConfig())
    
    # RL environment settings
    RL_CONFIG_PATH = "/scratch/dc4971/nocturne/experiments/human_regularized/rl_config.yaml"

    # Set path to trained IL model from wandb (make sure action and obs spaces match)
    BC_MODEL_VERSION = "v13"
    ARTIFACT_PATH = f"daphnecor/behavioral_cloning_discrete/bc_model_grid:{BC_MODEL_VERSION}"

    # Names
    SWEEP_NAME = "hr_ppo_experiments"
    MODEL_TYPE = "hr_ppo_model"
    SCENE_NAME = "_intersection_2agents" # example_scenario
    MAX_AGENTS = 2 #TODO
    NUM_INDEP_RUNS = 1
    SAVE_MODEL_FREQ = 50 # Save model every x iterations

    # Load RL config
    with open(RL_CONFIG_PATH, "r") as stream:
        RL_ENV_ARGS = yaml.safe_load(stream)

    # Load valid vehicle files
    with open(os.path.join(BASE_PATH, "valid_files.json")) as f:
        valid_veh_dict = json.load(f)

    # Changed to action space
    RL_ENV_ARGS["accel_discretization"] = 5
    RL_ENV_ARGS["accel_lower_bound"] = -2
    RL_ENV_ARGS["accel_upper_bound"] = 2

    RL_ENV_ARGS["steering_discretization"] = 5
    RL_ENV_ARGS["steering_lower_bound"] = -0.7
    RL_ENV_ARGS["steering_upper_bound"] = 0.7

    # Define the search space
    sweep_configuration = {  
        'method': 'random',  
        'metric': {'goal': 'minimize', 'name': 'loss'},  
        'parameters': {  
            'num_rollouts': { 'values': [80]},             
            'total_iters': {'values': [20]},                
            'learning_rate': { 'values': [5e-5, 1e-4]},  
            'ent_coef': { 'values': [0, 0.05]},                   
            'vf_coef': { 'values': [0.1, 0.005]},                    
            'hidden_layers': {'values': [[4096, 2048, 1024, 512, 128]]}, 
            'lambda_hr': {'values': [0, 0.1, 0.2, 0.8]},                    
        }  
    }

    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project=SWEEP_NAME,
    )

    # Start sweep job!
    wandb.agent(sweep_id, function=main, count=NUM_INDEP_RUNS)