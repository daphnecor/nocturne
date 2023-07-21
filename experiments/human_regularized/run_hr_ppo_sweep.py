import datetime
import glob
import logging
import random
import time
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import utils
import yaml
import wandb
import scipy

from base_env import BaseEnv
from constants import (
    HumanPolicyConfig,
    PPOExperimentConfig,
    WandBSettings,
)

from rl_models import Agent
from bc_models import BehavioralCloningAgentJoint
from nocturne import Action

from dataclasses import asdict

# Set wandb waiting time
os.environ["WANDB__SERVICE_WAIT"] = "300"

logging.basicConfig(level=logging.INFO)

def scipy_kl_batch(tau_dist, actor_dist):  
    batch_size = tau_dist.probs.shape[0]
    sum_kl_div = 0
    for i in range(batch_size):
        p_tau = tau_dist.probs[i, :].detach().numpy()
        p_actor = actor_dist.probs[i, :].detach().numpy()
        sum_kl_div += scipy.special.kl_div(p_tau, p_actor).sum()
    
    avg_kl_div = sum_kl_div / batch_size
    return avg_kl_div

def main():

    # Default configurations (params we're not tuning)
    args_exp = PPOExperimentConfig()
    args_wandb = WandBSettings()
    args_rl_env = utils.load_yaml_file(RL_SETTINGS_PATH)
    combined_dict = {**args_rl_env, **asdict(args_exp)}

    # Initialize run
    run = wandb.init()
    artifact = wandb.Artifact(name='ppo_network', type='model')
    rl_env_artifact = wandb.Artifact(name='rl_settings', type='config')
    rl_env_artifact.add_file(RL_SETTINGS_PATH)
    run.log_artifact(rl_env_artifact)

    # Use BC model
    bc_model_artifact = run.use_artifact(ARTIFACT_PATH, type="model")
    bc_model_artifact_dir = bc_model_artifact.download()
    checkpoint = torch.load(
        f"{bc_model_artifact_dir}/BC_model.pt", map_location=torch.device("cpu")
    )

    logging.info(f'Loading imitation learning model...')

    # Get sweep params
    NUM_ROLLOUTS = wandb.config.num_rollouts
    TOTAL_ITERS = wandb.config.total_iters
    LR = wandb.config.learning_rate
    LAMBDA = wandb.config.lambda_hr
    ENT_COEF = wandb.config.ent_coef
    VF_COEF = wandb.config.vf_coef
    COLL_PENALTY = wandb.config.collision_penalty

    # Log
    now = datetime.datetime.now()
    formatted_time = now.strftime("%D%H%M")
    run_name = f"Nocturne-v0__{args_wandb.exp_name}_{formatted_time}"

    # Seed
    random.seed(args_exp.seed)
    np.random.seed(args_exp.seed)
    torch.manual_seed(args_exp.seed)
    torch.backends.cudnn.deterministic = args_exp.torch_deterministic

    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args_exp.cuda else "cpu"
    )
    logging.critical(f"DEVICE: {device}")

    # Make environment 
    env = BaseEnv(args_rl_env)
    env.collision_penalty = COLL_PENALTY

    logging.info(f'collision penalty: {env.collision_penalty}')

    # Initialize actor and critic models
    obs_space_dim = env.observation_space.shape[0]
    act_space_dim = env.action_space.n
    ppo_agent = Agent(obs_space_dim, act_space_dim).to(device)
    
    # Optimizer 
    optimizer = optim.Adam(ppo_agent.parameters(), lr=LR, eps=1e-5)
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    # Load human anchor policy
    human_anchor_policy = BehavioralCloningAgentJoint(
        num_states=obs_space_dim,
        hidden_layers=checkpoint["hidden_layers"],
        actions_discretizations=checkpoint["actions_discretizations"],
        actions_bounds=checkpoint["actions_bounds"],
        device=device,
        deterministic=True, #TODO: check
    ).to(device)
    human_anchor_policy.load_state_dict(checkpoint["model_state_dict"])
    human_anchor_policy.eval()

    logging.info(f'actor_critic obs space: {obs_space_dim} | human policy obs_space: {obs_space_dim}')

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
        last_step = []
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

                # Compute kl divergence to the human anchor policy
                action_idx, tau_dist = human_anchor_policy(b_obs[mb_inds])
                actor_dist = ppo_agent.get_policy(b_obs[mb_inds])
                
                kl_div_to_human_policy = kl_loss(tau_dist.probs.log(), actor_dist.probs)
                # Use below to verify correctness, pytorch implementation is faster
                #kl_div_scipy = scipy_kl_batch(actor_dist, tau_dist)

                # Check if the minibatch has at least two elements
                if len(mb_inds) < 2:
                    continue  # Skip this minibatch 

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
                    LAMBDA * (kl_div_to_human_policy) + 
                    (1 - LAMBDA) * (pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss)
                )
                
                # Backwards
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
                    "charts/b_advantages_mean": b_advantages.mean(),
                    "charts/entropy_human_policy_mean": tau_dist.entropy().detach().cpu().numpy().mean(),
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "losses/kl_div_to_human_policy": kl_div_to_human_policy.item(),
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy_ppo_agent_mean": entropy_loss.item(),
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
            model_path = os.path.join(wandb.run.dir, f"hr_ppo_model_{SCENE_NAME}.pt")
            torch.save(
                obj={
                    "iter": iter_,
                    "model_state_dict": ppo_agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "obs_space_dim": obs_space_dim,
                    "act_space_dim": act_space_dim,
                    "iter": iter_,
                    "policy_loss": pg_loss,
                    "lambda_hr": LAMBDA,
                    "kl_div_to_human_policy": kl_div_to_human_policy,
                    "ep_reward": current_ep_reward,
                    "minibatch_size": minibatch_size,
                },
                f=model_path,
            )
            logging.info(f"\nSaved model at {model_path}")
        
    # Save trained PPO agent as an artifact
    artifact.add_file(local_path=model_path)
    run.log_artifact(artifact)

    # # # #     End of run    # # # #
    env.close()


if __name__ == "__main__":

    # Path to best imitation learning model so far
    BC_MODEL_VERSION = "v13"
    ARTIFACT_PATH = f"daphnecor/behavioral_cloning_grid_actions/bc_model_grid:{BC_MODEL_VERSION}"

    SCENE_NAME = "simple_intersection"
    MAX_AGENTS = 2 #TODO: extend this to work with n agents
    RL_SETTINGS_PATH = "/scratch/dc4971/nocturne/experiments/human_regularized/rl_config.yaml"
    PROJECT_NAME = "hr_ppo_sweeps"
    NUM_INDEP_RUNS = 2
    SAVE_MODEL_FREQ = 15

    # Define the search space
    sweep_configuration = {  
        'method': 'random',  
        'metric': {'goal': 'minimize', 'name': 'loss'},  
        'parameters': {  
            'collision_penalty': { 'values': [0]}, 
            'num_rollouts': { 'values': [90]},                  # Batch size (rollouts per iteration)
            'total_iters': {'values': [100]},                  # Total number of iterations
            'learning_rate': { 'values': [5e-5, 1e-4, 5e-4]},  # Learning rate
            'ent_coef': { 'values': [0.0]},                    # Entropy coefficient
            'vf_coef': { 'values': [0.5, 0.25]},               # Value function coefficient
            'lambda_hr': {'values': [1]},                      # HR coefficient
        }  
    } 

    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project=PROJECT_NAME,
    )

    # Start sweep job!
    wandb.agent(sweep_id, function=main, count=NUM_INDEP_RUNS)