import datetime
import glob
import logging
import random
import time
import os
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import utils
import yaml
import wandb
from pprint import pprint

import nocturne_gym as gym
from constants import PPOExperimentConfig, WandBSettings
from rl_models import PPOAgent
from nocturne import Action

from dataclasses import asdict, dataclass

# Set wandb waiting time to avoid cluster timeout errors
os.environ["WANDB__SERVICE_WAIT"] = "300"

logging.basicConfig(level=logging.INFO)

def main():

    # Default configurations to be stored as metadata
    args_exp = PPOExperimentConfig()
    args_wandb = WandBSettings()
    meta_data_dict = {**args_rl_env, **asdict(args_exp)}

    # Initialize run
    run = wandb.init()

    # Get sweep params
    NUM_ROLLOUTS = wandb.config.num_rollouts
    TOTAL_ITERS = wandb.config.total_iters
    LR = wandb.config.learning_rate
    ENT_COEF = wandb.config.ent_coef
    VF_COEF = wandb.config.vf_coef
    COLL_PENALTY = wandb.config.collision_penalty
    HIDDEN_LAYERS = wandb.config.hidden_layers

    # Seed
    random.seed(args_exp.seed)
    np.random.seed(args_exp.seed)
    torch.manual_seed(args_exp.seed)
    torch.backends.cudnn.deterministic = args_exp.torch_deterministic

    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args_exp.cuda else "cpu"
    )
    logging.info(f"DEVICE: {device}")

    # Make environment from selected traffic scene
    env = gym.NocturneEnv(
        path_to_scene=path_to_file,
        scene_name=file,
        valid_veh_dict=valid_veh_dict,
        scenario_config=scenario_config,
        cfg=args_rl_env,
    )
    env.collision_penalty = COLL_PENALTY

    logging.info(f'env action space:')
    pprint(env.idx_to_actions)

    logging.info(f'collision penalty: {env.collision_penalty}')

    # Initialize actor and critic models
    obs_space_dim = env.observation_space.shape[0]
    act_space_dim = env.action_space.n
    ppo_agent = PPOAgent(
        obs_space_dim,
        act_space_dim,
        hidden_layers=HIDDEN_LAYERS,

    ).to(device)

    # Create descriptive model name
    MODEL_NAME = f"ppo_model_Dstate_{obs_space_dim}_Dact_{act_space_dim}_S{SCENE_NAME}"
    model_path = os.path.join(wandb.run.dir, f"{MODEL_NAME}.pt")

    # Optimizer 
    optimizer = optim.Adam(ppo_agent.parameters(), lr=LR, eps=1e-5)

    global_step = 0

    for iter_ in range(1, TOTAL_ITERS):
        start_iter = time.time()

        # Storage
        obs_tensor = torch.zeros(
            (
                NUM_ROLLOUTS,
                args_exp.num_steps,
                MAX_AGENTS,
                obs_space_dim,
            )
        ).to(device)
        rew_tensor = torch.zeros(
            (NUM_ROLLOUTS, args_exp.num_steps, MAX_AGENTS)
        ).to(device)
        act_tensor = torch.zeros(
            (NUM_ROLLOUTS, args_exp.num_steps, MAX_AGENTS)
        ).to(device)
        done_tensor = torch.zeros(
            (NUM_ROLLOUTS, args_exp.num_steps, MAX_AGENTS)
        ).to(device)
        logprob_tensor = torch.zeros(
            (NUM_ROLLOUTS, args_exp.num_steps, MAX_AGENTS)
        ).to(device)
        value_tensor = torch.zeros(
            (NUM_ROLLOUTS, args_exp.num_steps, MAX_AGENTS)
        ).to(device)
        veh_coll_tensor = torch.zeros((NUM_ROLLOUTS)).to(device)
        edge_coll_tensor = torch.zeros((NUM_ROLLOUTS)).to(device)
        goal_tensor = torch.zeros((NUM_ROLLOUTS)).to(device)

        # # # #  Collect experience with current policy  # # # #
        start_rollouts = time.time()
       
        for rollout_step in range(NUM_ROLLOUTS):
            # Reset environment
            # NOTE: this can either be the same env or a new traffic scene
            # currently using the same scene for debugging purposes
            next_obs_dict = env.reset()

            controlled_agents = [agent.getID() for agent in env.controlled_vehicles]
            num_agents = len(controlled_agents)
            dict_next_done = {agent_id: False for agent_id in controlled_agents}
            already_done_ids = env.done_ids.copy()
            current_ep_reward = 0
            current_ep_rew_agents = {agent_id: 0 for agent_id in controlled_agents}

            # Set data buffer for within scene logging
            buffer = utils.RolloutBuffer(
                controlled_agents,
                args_exp.num_steps,
                obs_space_dim,
                act_space_dim,
                device,
            )

            # Adapt learning rate based on how far we are in the learning process
            if args_exp.anneal_lr:
                frac = 1.0 - (iter_ - 1.0) / TOTAL_ITERS
                lrnow = frac * args_exp.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

    
            # # # #  Interact with environment  # # # #
            start_env = time.time()
            for step in range(0, args_exp.num_steps):

                global_step += 1 * num_agents

                # Store dones and observations for active agents
                for agent_id in controlled_agents:
                    buffer.dones[agent_id][step] = dict_next_done[agent_id] * 1
                    if agent_id not in already_done_ids:
                        buffer.observations[agent_id][step, :] = torch.Tensor(
                            next_obs_dict[agent_id]
                        )
                    else:
                        continue

                # Use policy network to select an action for every agent based
                # on current observation
                with torch.no_grad():
                    action_dict = {
                        agent_id: None
                        for agent_id in controlled_agents
                        if agent_id not in already_done_ids
                    }

                    for agent_id in action_dict.keys():
                        action, logprob, _, value = ppo_agent.get_action_and_value(
                            torch.Tensor(next_obs_dict[agent_id]).to(device)
                        )
                        # Store in buffer
                        buffer.values[agent_id][step] = value.flatten()
                        buffer.actions[agent_id][step] = action
                        buffer.logprobs[agent_id][step] = logprob

                        # Store in action_dict
                        action_dict[agent_id] = action.item()

                # Take simultaneous action in env
                next_obs_dict, reward_dict, next_done_dict, info_dict = env.step(
                    action_dict
                )

                # Store rewards
                for agent_id in next_obs_dict.keys():
                    buffer.rewards[agent_id][step] = torch.from_numpy(
                        np.asarray(reward_dict[agent_id])
                    ).to(device)

                # Update collisions and whether goal is achieved
                veh_coll_tensor[rollout_step] += sum(
                    info["veh_veh_collision"] for info in info_dict.values()
                )
                edge_coll_tensor[rollout_step] += sum(
                    info["veh_edge_collision"] for info in info_dict.values()
                )
                goal_tensor[rollout_step] += sum(
                    info["goal_achieved"] for info in info_dict.values()
                )

                # Update episodal rewards
                current_ep_reward += sum(reward_dict.values())

                # Update done agents
                for agent_id in next_obs_dict.keys():
                    current_ep_rew_agents[agent_id] += reward_dict[agent_id].item()
                    if next_done_dict[agent_id]:
                        dict_next_done[agent_id] = True

                already_done_ids = [
                    agent_id for agent_id, value in dict_next_done.items() if value
                ]

                # End the game early if all agents are done
                if len(already_done_ids) == num_agents or step == args_exp.num_steps:
                    break

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

            # Store rollout scene experience
            obs_tensor[rollout_step, :, :num_agents, :] = utils.dict_to_tensor(
                buffer.observations
            ).to(device)
            rew_tensor[rollout_step, :, :num_agents] = utils.dict_to_tensor(
                buffer.rewards
            ).to(device)
            act_tensor[rollout_step, :, :num_agents] = utils.dict_to_tensor(
                buffer.actions
            ).to(device)
            done_tensor[rollout_step, :, :num_agents] = utils.dict_to_tensor(buffer.dones).to(
                device
            ).to(device)
            logprob_tensor[rollout_step, :, :num_agents] = utils.dict_to_tensor(
                buffer.logprobs
            ).to(device)
            value_tensor[rollout_step, :, :num_agents] = utils.dict_to_tensor(
                buffer.values
            ).to(device)

            # Normalize counts by agents in scene
            veh_coll_tensor[rollout_step] /= num_agents
            edge_coll_tensor[rollout_step] /= num_agents
            goal_tensor[rollout_step] /= num_agents

            # Logging
            if args_wandb.track:
                wandb.log(
                    {
                        "global_step": global_step,
                        "global_iter": iter_,
                        "charts/num_agents_in_scene": num_agents,
                        "charts/total_episodic_return": current_ep_reward,
                        "charts/close_agent(3)_episodic_return": current_ep_rew_agents[3],
                        "charts/far_agent(32)_episodic_return": current_ep_rew_agents[32],
                        "charts/episodic_length": step,
                        "charts/goal_achieved_rate": goal_tensor[rollout_step],
                        "charts/veh_veh_collision_rate": veh_coll_tensor[rollout_step],
                        "charts/veh_edge_collision_rate": edge_coll_tensor[
                            rollout_step
                        ],
                    }
                )
                

            # Clear buffer for next scene
            buffer.clear()

        # # # # # # END ROLLOUTS # # # # # #
        time_rollouts = time.time() - start_rollouts

        # # # # Compute advantage estimate via GAE on collected experience # # # #
        # Flatten along the num_agents x policy rollout steps axes
        dones = done_tensor.permute(0, 2, 1).reshape(
            num_agents * NUM_ROLLOUTS, args_exp.num_steps
        ).to(device)
        values = value_tensor.permute(0, 2, 1).reshape(
            num_agents * NUM_ROLLOUTS, args_exp.num_steps
        ).to(device)
        rewards = rew_tensor.permute(0, 2, 1).reshape(
            num_agents * NUM_ROLLOUTS, args_exp.num_steps
        ).to(device)
        logprobs = logprob_tensor.permute(0, 2, 1).reshape(
            num_agents * NUM_ROLLOUTS, args_exp.num_steps
        ).to(device)
        actions = act_tensor.permute(0, 2, 1).reshape(
            num_agents * NUM_ROLLOUTS, args_exp.num_steps
        ).to(device)

        obs_flat = obs_tensor.permute(0, 2, 1, 3).reshape(
            num_agents * NUM_ROLLOUTS,
            args_exp.num_steps,
            obs_space_dim,
        ).to(device)
        dones_flat = done_tensor.permute(0, 2, 1).reshape(
            num_agents * NUM_ROLLOUTS, args_exp.num_steps
        ).to(device)

        # Select the most recent observation for every policy rollout and agent
        last_step_alive = utils.find_last_zero_index(dones)
        next_obs = torch.stack(
            [
                obs_flat[idx, last_step_alive[idx], :]
                for idx in range(len(last_step_alive))
            ]
        ).to(device)

        # Assumption: all agents are done after 80 steps, even if they haven't
        # reached their target
        next_done = torch.ones(size=(num_agents * NUM_ROLLOUTS,)).to(device)

        with torch.no_grad():
            next_value = ppo_agent.get_value(next_obs).reshape(-1).to(device)
            advantages = torch.zeros(
                (MAX_AGENTS * NUM_ROLLOUTS, args_exp.num_steps)
            ).to(device)
            lastgaelam = 0
            for t in reversed(range(args_exp.num_steps)):
                if t == args_exp.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[:, t + 1]
                    nextvalues = values[:, t + 1]

                # Compute TD-error
                delta = (
                    rewards[:, t]
                    + args_exp.gamma * nextvalues * nextnonterminal
                    - values[:, t]
                )
                # Update advantage for timestep
                lastgaelam = (
                    delta
                    + args_exp.gamma
                    * args_exp.gae_lambda
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
        if args_wandb.track: wandb.log({"batch_size": batch_size})
        start_optim = time.time()

        # # # #  Optimization   # # # #
        for epoch in range(args_exp.update_epochs):

            # Shuffle batch indices
            np.random.shuffle(b_inds)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # Check if the minibatch has at least two elements
                if len(mb_inds) < 2:
                    continue  # Skip this minibatch and move to the next one

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
                        ((ratio - 1.0).abs() > args_exp.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args_exp.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args_exp.clip_coef, 1 + args_exp.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args_exp.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args_exp.clip_coef,
                        args_exp.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = (
                    pg_loss
                    - ENT_COEF * entropy_loss
                    + VF_COEF * v_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ppo_agent.parameters(), args_exp.max_grad_norm)
                optimizer.step()

            if args_exp.target_kl is not None:
                if approx_kl > args_exp.target_kl:
                    break

        # # # #     End of iteration     # # # #

        # Logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        explained_var = np.nan if np.var(y_true) == 0 else 1 - (np.var(y_true - y_pred)) / np.var(y_true)

        if args_wandb.track:
            wandb.log(
                {
                    "global_step": global_step,
                    "charts/b_advantages": wandb.Histogram(b_advantages.cpu().numpy()),
                    "charts/b_advantages_mean": b_advantages.mean(),
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
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

        # Save model checkpoint in wandb directory
        if iter_ % SAVE_MODEL_FREQ == 0:

            # Create model artifact
            model_artifact = wandb.Artifact(
                name=f"{MODEL_ARTIFACT_NAME}_iter_{iter_}", 
                type=MODEL_TYPE,
                description=f"PPO on scene: {SCENE_NAME}",
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
                    "ep_reward": current_ep_reward,
                    "minibatch_size": minibatch_size,
                },
                f=model_path,
            )

            # Save model artifact  
            model_artifact.add_file(local_path=model_path)
            wandb.save(model_path, base_path=wandb.run.dir)
            run.log_artifact(model_artifact)

            logging.info(f"Stored {MODEL_ARTIFACT_NAME} after {iter_} iters.")

    # # # #     End of run    # # # #
    env.close()


if __name__ == "__main__":

    scenario_config = {
        'start_time': 0, 
        'allow_non_vehicles': False, 
        'moving_threshold': 0.2, 
        'speed_threshold': 0.05, 
        'max_visible_road_points': 500, 
        'sample_every_n': 1, 
        'road_edge_first': False
    }
    
    MODEL_ARTIFACT_NAME = 'ppo_network'
    MODEL_TYPE = 'ppo_model'
    file = "example_scenario.json"
    path_to_file = "/scratch/dc4971/nocturne/data/formatted_json_v2_no_tl_train/"
    SCENE_NAME = "_intersection_2agents" # example_scenario
    MAX_AGENTS = 2 #TODO
    RL_SETTINGS_PATH = "/scratch/dc4971/nocturne/experiments/human_regularized/rl_config.yaml"
    SWEEP_NAME = "ppo_sweeps"
    NUM_INDEP_RUNS = 3
    SAVE_MODEL_FREQ = 100 # Save model every x iterations

    # Load RL config
    with open(RL_SETTINGS_PATH, "r") as stream:
        args_rl_env = yaml.safe_load(stream)

    # Load files
    with open(os.path.join(path_to_file, "valid_files.json")) as f:
        valid_veh_dict = json.load(f)

    # Adapt action space #TODO
    args_rl_env["accel_discretization"] = 3
    args_rl_env["accel_lower_bound"] = 0
    args_rl_env["accel_upper_bound"] = 2

    args_rl_env["steering_discretization"] = 3
    args_rl_env["steering_lower_bound"] = -1
    args_rl_env["steering_upper_bound"] = 1

    # Define the search space
    sweep_configuration = {  
        'method': 'random',  
        'metric': {'goal': 'minimize', 'name': 'loss'},  
        'parameters': {  
            'collision_penalty': { 'values': [50]}, 
            'num_rollouts': { 'values': [80, 90]},             
            'total_iters': {'values': [1000]},                
            'learning_rate': { 'values': [5e-5, 1e-4, 5e-4]},  
            'ent_coef': { 'values': [0, 0.05]},                   
            'vf_coef': { 'values': [0.005]},                    
            'hidden_layers': {'values': [[4096, 2048, 1024, 512, 128]]}, 
        }  
    }

    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project=SWEEP_NAME,
    )

    # Start sweep job!
    wandb.agent(sweep_id, function=main, count=NUM_INDEP_RUNS)