from copy import copy, deepcopy
import datetime
import glob
import logging
from multiprocessing import Pool
import random
import time
from typing import Any, Dict
import os
from pathlib import Path

import numpy as np
import torch
# import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


import utils
import yaml
import wandb

from base_env import BaseEnv
from constants import (
    HumanPolicyConfig,
    PPOExperimentConfig,
    WandBSettings,
)

from rl_models import Agent
from cfgs.config import set_display_window
from nocturne import Action

from dataclasses import asdict, dataclass

logging.basicConfig(level=logging.DEBUG)

RL_SETTINGS_PATH = "experiments/human_regularized/rl_config.yaml"


def _do_policy_rollout(
    rollout_step: int,
    args_exp: PPOExperimentConfig,
    Env: BaseEnv,
    args_rl_env: Dict[str, Any], # TODO: Add type
    ppo_agent: Agent,
):
    print(f"Doing policy rollout {rollout_step + 1} from process {os.getpid()}")
    rollout_start = time.perf_counter()
    # # Reset environment
    # # NOTE: this can either be the same env or a new traffic scene
    # # currently using the same scene for debugging purposes
    # next_obs_dict_list = [env.reset() for env in env_list]
    env = Env(args_rl_env)
    next_obs_dict = env.reset()

    controlled_agents = [agent.getID() for agent in env.controlled_vehicles]
    dict_next_done = {agent_id: False for agent_id in controlled_agents}
    already_done_ids = env.done_ids.copy()
    # current_ep_reward = 0
    # current_ep_rew_agents = {agent_id: 0 for agent_id in controlled_agents}

    # Set data buffer for within scene logging
    rollout_buffer = utils.RolloutBuffer(
        controlled_agents,
        args_exp.num_steps,
        env.observation_space.shape[0],
        env.action_space.n,
        "cpu", # args_exp.device,
    )

    # # # #  Interact with environment  # # # #
    for step in range(0, args_exp.num_steps):
        # Store dones and observations for active agents
        for agent_id in controlled_agents:
            rollout_buffer.dones[agent_id][step] = dict_next_done[agent_id] * 1
            if agent_id not in already_done_ids:
                rollout_buffer.observations[agent_id][step, :] = torch.Tensor(
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
                    torch.Tensor(next_obs_dict[agent_id])
                )
                # Store in buffer
                rollout_buffer.values[agent_id][step] = value.flatten()
                rollout_buffer.actions[agent_id][step] = action
                rollout_buffer.logprobs[agent_id][step] = logprob

                # Store in action_dict
                action_dict[agent_id] = action.item()

        # Take simultaneous action in env
        next_obs_dict, reward_dict, next_done_dict, info_dict = env.step(
            action_dict
        )

        # Store rewards
        for agent_id in next_obs_dict.keys():
            rollout_buffer.rewards[agent_id][step] = torch.from_numpy(
                np.asarray(reward_dict[agent_id])
            )

        # Update done agents
        for agent_id in next_obs_dict.keys():
            # current_ep_rew_agents[agent_id] += reward_dict[agent_id].item()
            if next_done_dict[agent_id]:
                dict_next_done[agent_id] = True

        already_done_ids = [
            agent_id for agent_id, value in dict_next_done.items() if value
        ]

        # End the game early if all agents are done
        if len(already_done_ids) == args_exp.max_agents or step == args_exp.num_steps:
            break

    rollout_stop = time.perf_counter()
    print(f"Rollout {rollout_step + 1} from process {os.getpid()} took {rollout_stop - rollout_start:.2f} seconds")

    return rollout_buffer

def main():
    # Configs
    args_exp = PPOExperimentConfig()
    args_wandb = WandBSettings()
    args_human_pol = HumanPolicyConfig()
    args_rl_env = utils.load_yaml_file(RL_SETTINGS_PATH)

    combined_dict = {**args_rl_env, **asdict(args_exp)}

    # Log
    now = datetime.datetime.now()
    formatted_time = now.isoformat(timespec="minutes")
    run_name = f"Nocturne-v0__{args_wandb.exp_name}_{formatted_time}"

    if args_wandb.track:
        run = wandb.init(
            project=args_wandb.project_name,
            config=combined_dict,
            group=args_wandb.group,
            name=run_name,
            save_code=True,
        )

    # Seed
    random.seed(args_exp.seed)
    np.random.seed(args_exp.seed)
    torch.manual_seed(args_exp.seed)
    torch.backends.cudnn.deterministic = args_exp.torch_deterministic

    # Set device
    args_exp.cuda = args_exp.cuda and torch.cuda.is_available()
    args_exp.device = torch.device("cuda" if args_exp.cuda else "cpu")
    logging.critical("DEVICE: %s", args_exp.device)

    # Environment setup
    env = BaseEnv(args_rl_env)

    obs_space_dim = env.observation_space.shape[0]
    act_space_dim = env.action_space.n

    # Initialize actor and critic models
    ppo_agent = Agent(obs_space_dim, act_space_dim).to(args_exp.device)
    optimizer = optim.Adam(ppo_agent.parameters(), lr=args_exp.learning_rate, eps=1e-5)

    for iter_ in range(1, args_exp.total_iters + 1):
        start_iter = time.time()

        logging.debug(f'Iteration: {iter_}')

        # Adapt learning rate based on how far we are in the learning process
        if args_exp.anneal_lr:
            frac = 1.0 - (iter_ - 1.0) / args_exp.total_iters
            lrnow = frac * args_exp.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Storage
        obs_tensor = torch.zeros(
            (
                args_exp.num_policy_rollouts,
                args_exp.num_steps,
                args_exp.max_agents,
                obs_space_dim,
            )
        ).to(args_exp.device)
        rew_tensor = torch.zeros(
            (args_exp.num_policy_rollouts, args_exp.num_steps, args_exp.max_agents)
        ).to(args_exp.device)
        act_tensor = torch.zeros(
            (args_exp.num_policy_rollouts, args_exp.num_steps, args_exp.max_agents)
        ).to(args_exp.device)
        done_tensor = torch.zeros(
            (args_exp.num_policy_rollouts, args_exp.num_steps, args_exp.max_agents)
        ).to(args_exp.device)
        logprob_tensor = torch.zeros(
            (args_exp.num_policy_rollouts, args_exp.num_steps, args_exp.max_agents)
        ).to(args_exp.device)
        value_tensor = torch.zeros(
            (args_exp.num_policy_rollouts, args_exp.num_steps, args_exp.max_agents)
        ).to(args_exp.device)
        veh_coll_tensor = torch.zeros((args_exp.num_policy_rollouts)).to(args_exp.device)
        edge_coll_tensor = torch.zeros((args_exp.num_policy_rollouts)).to(args_exp.device)
        goal_tensor = torch.zeros((args_exp.num_policy_rollouts)).to(args_exp.device)

        # # # #  Collect experience with current policy  # # # #
        start_rollouts = time.time()
        last_step = []

        start_rollout = time.time()
        
        ppo_agent.to("cpu")

        # mp.set_start_method('spawn')
        with Pool(args_exp.num_processes) as pool:
            rollout_buffers = pool.starmap(
                _do_policy_rollout,
                [
                    (
                        rollout_step,
                        copy(args_exp),
                        BaseEnv,
                        args_rl_env.copy(),
                        deepcopy(ppo_agent),
                    )
                    for rollout_step in range(args_exp.num_policy_rollouts)
                ]
            )
        ppo_agent.to(args_exp.device)

        logging.info(f'Total time rollouts: {time.time() - start_rollout}')

        for rollout_step, rollout_buffer in enumerate(rollout_buffers):
            # Store rollout scene experience
            obs_tensor[rollout_step, :, :args_exp.max_agents, :] = utils.dict_to_tensor(
                rollout_buffer.observations
            ).to(args_exp.device)
            rew_tensor[rollout_step, :, :args_exp.max_agents] = utils.dict_to_tensor(
                rollout_buffer.rewards
            ).to(args_exp.device)
            act_tensor[rollout_step, :, :args_exp.max_agents] = utils.dict_to_tensor(
                rollout_buffer.actions
            ).to(args_exp.device)
            done_tensor[rollout_step, :, :args_exp.max_agents] = utils.dict_to_tensor(rollout_buffer.dones).to(
                args_exp.device
            ).to(args_exp.device)
            logprob_tensor[rollout_step, :, :args_exp.max_agents] = utils.dict_to_tensor(
                rollout_buffer.logprobs
            ).to(args_exp.device)
            value_tensor[rollout_step, :, :args_exp.max_agents] = utils.dict_to_tensor(
                rollout_buffer.values
            ).to(args_exp.device)

        # # # # # # END ROLLOUTS # # # # # #
        time_rollouts = time.time() - start_rollouts

        # # # # Compute advantage estimate via GAE on collected experience # # # #
        # Flatten along the args_exp.max_agents x policy rollout steps axes
        dones = done_tensor.permute(0, 2, 1).reshape(
            args_exp.max_agents * args_exp.num_policy_rollouts, args_exp.num_steps
        ).to(args_exp.device)
        values = value_tensor.permute(0, 2, 1).reshape(
            args_exp.max_agents * args_exp.num_policy_rollouts, args_exp.num_steps
        ).to(args_exp.device)
        rewards = rew_tensor.permute(0, 2, 1).reshape(
            args_exp.max_agents * args_exp.num_policy_rollouts, args_exp.num_steps
        ).to(args_exp.device)
        logprobs = logprob_tensor.permute(0, 2, 1).reshape(
            args_exp.max_agents * args_exp.num_policy_rollouts, args_exp.num_steps
        ).to(args_exp.device)
        actions = act_tensor.permute(0, 2, 1).reshape(
            args_exp.max_agents * args_exp.num_policy_rollouts, args_exp.num_steps
        ).to(args_exp.device)

        obs_flat = obs_tensor.permute(0, 2, 1, 3).reshape(
            args_exp.max_agents * args_exp.num_policy_rollouts,
            args_exp.num_steps,
            obs_space_dim,
        ).to(args_exp.device)
        dones_flat = done_tensor.permute(0, 2, 1).reshape(
            args_exp.max_agents * args_exp.num_policy_rollouts, args_exp.num_steps
        ).to(args_exp.device)

        # Select the most recent observation for every policy rollout and agent
        last_step_alive = utils.find_last_zero_index(dones)
        next_obs = torch.stack(
            [
                obs_flat[idx, last_step_alive[idx], :]
                for idx in range(len(last_step_alive))
            ]
        ).to(args_exp.device)

        # Assumption: all agents are done after 80 steps, even if they haven't
        # reached their target
        next_done = torch.ones(size=(args_exp.max_agents * args_exp.num_policy_rollouts,)).to(args_exp.device)

        with torch.no_grad():
            next_value = ppo_agent.get_value(next_obs).reshape(-1).to(args_exp.device)
            advantages = torch.zeros(
                (args_exp.max_agents * args_exp.num_policy_rollouts, args_exp.num_steps)
            ).to(args_exp.device)
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
        minibatch_size = batch_size // args_exp.max_agents
        if args_wandb.track: wandb.log({"batch_size": batch_size})
        start_optim = time.time()

        # # # #  Optimization   # # # #
        for epoch in range(args_exp.update_epochs):
            # Shuffle batch indices
            np.random.shuffle(b_inds)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # COMPUTE KL DIV TO HUMAN ANCHOR POLICY TODO
                #action, log_prob, tau_dist = human_anchor_policy(b_obs[mb_inds])
                actor_dist = ppo_agent.get_policy(b_obs[mb_inds])
                # kl_div = kl_loss(tau_dist.probs, actor_dist.probs)

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
                    - args_exp.ent_coef * entropy_loss
                    + args_exp.vf_coef * v_loss
                    # - args_exp.human_kl_lam * kl_div
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
        if iter_ % 25 == 0 and args_wandb.track and args_exp.save_model:
            model_path = os.path.join(wandb.run.dir, f"Nocturne-v0_ppo.pt")
            torch.save(
                obj={
                    "iter": iter_,
                    "model_state_dict": ppo_agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "policy_loss": pg_loss,
                    "ep_reward": current_ep_reward,
                    "minibatch_size": minibatch_size,
                },
                f=model_path,
            )
            logging.critical(f"\nSaved model at {model_path}")


    # # # #     End of run    # # # #
    env.close()


if __name__ == "__main__":
    main()
