import argparse
import os
import random
import time
from distutils.util import strtobool
from dataclasses import dataclass, asdict

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import wandb
#import pdb
from constants import PPOExperimentConfig, NocturneConfig, WandBSettings, HumanPolicyConfig

import yaml
from base_env import BaseEnv
import logging
from imit_models import BehavioralCloningAgentJoint

logging.basicConfig(level=logging.CRITICAL)


def make_env(config_path, seed, run_name):
    """Make nocturne environment."""
    with open(config_path, "r") as stream:
        base_env_config = yaml.safe_load(stream)
    env = BaseEnv(base_env_config)
    env.seed(seed)
    return env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def dict_to_tensor(my_dict):
    tensor_list = []
    for agent_id, tensor in my_dict.items():
        tensor_list.append(torch.Tensor(tensor))
    stacked_tensor = torch.stack(tensor_list, dim=1)
    return stacked_tensor.squeeze()


class Agent(nn.Module):
    def __init__(self, envs, state_dim, action_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_policy(self, x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return probs


class RolloutBuffer:
    """
    Shared rollout buffer to store collected trajectories for every agent we control.
    Must be reset after the policy network is updated.
    """

    def __init__(
        self, controlled_agents, num_steps, obs_space_dim, act_space_dim, device
    ):
        """
        Args:
            controlled_vehicles (list[nocturne vehicles])
        """
        self.observations = self.create_tensor_dict(
            controlled_agents, num_steps, device, obs_space_dim
        )
        self.actions = self.create_tensor_dict(
            controlled_agents,
            num_steps,
            device,
        )
        self.logprobs = self.create_tensor_dict(controlled_agents, num_steps, device)
        self.rewards = self.create_tensor_dict(controlled_agents, num_steps, device)
        self.dones = self.create_tensor_dict(controlled_agents, num_steps, device)
        self.values = self.create_tensor_dict(controlled_agents, num_steps, device)

    def create_tensor_dict(self, controlled_agents, num_steps, device="cpu", dim=None):
        tensor_dict = {}
        for agent in controlled_agents:
            key = agent
            if dim is not None:
                tensor = torch.zeros((num_steps, dim))
            else:
                tensor = torch.zeros((num_steps,))
            tensor_dict[key] = tensor.to(device)
        return tensor_dict

    def clear(self):
        for key in self.observations.keys():
            self.observations[key].zero_()
            self.actions[key].zero_()
            self.logprobs[key].zero_()
            self.rewards[key].zero_()
            self.dones[key].zero_()
            self.values[key].zero_()


if __name__ == "__main__":

    # Configs
    args_exp = PPOExperimentConfig()
    args_wandb = WandBSettings(
        track=True
    )
    args_env = NocturneConfig()
    args_human_pol = HumanPolicyConfig()

    run_name = f"{args_env.env_id}__{args_wandb.exp_name}"

    if args_wandb.track:
        wandb.init(
            project = args_wandb.project_name,
            sync_tensorboard = True,
            config = asdict(args_wandb),
            group = "nocturne",
            name = run_name,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args_exp).items()])),
    )

    # Seeding
    random.seed(args_exp.seed)
    np.random.seed(args_exp.seed)
    torch.manual_seed(args_exp.seed)
    torch.backends.cudnn.deterministic = args_exp.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args_exp.cuda else "cpu")

    # Env setup
    env = make_env(args_env.nocturne_rl_cfg, args_exp.seed, run_name)

    # State and action space dimension
    observation_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.n

    agent = Agent(env, observation_space_dim, action_space_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args_exp.learning_rate, eps=1e-5)
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    # Load human anchor policy
    human_anchor_policy = BehavioralCloningAgentJoint(
        num_inputs=observation_space_dim,
        config=args_human_pol,
        device=device,
    ).to(device)
    
    human_anchor_policy.load_state_dict(
        torch.load(args_human_pol.pretrained_model_path)
    )

    global_step = 0 
    start_time = time.time()
    MAX_AGENTS = 3

    for iter in range(1, args_exp.total_iters + 1):  

        writer.add_scalar("charts/iter", iter, global_step)

        # Storage
        obs_tensor = torch.zeros((args_exp.num_policy_rollouts, args_exp.num_steps, MAX_AGENTS, observation_space_dim,))
        rew_tensor = torch.zeros((args_exp.num_policy_rollouts, args_exp.num_steps, MAX_AGENTS))
        act_tensor = torch.zeros((args_exp.num_policy_rollouts, args_exp.num_steps, MAX_AGENTS))
        done_tensor = torch.zeros((args_exp.num_policy_rollouts, args_exp.num_steps, MAX_AGENTS))
        logprob_tensor = torch.zeros((args_exp.num_policy_rollouts, args_exp.num_steps, MAX_AGENTS))
        value_tensor = torch.zeros((args_exp.num_policy_rollouts, args_exp.num_steps, MAX_AGENTS))

        # # # #  Collect experience with current policy  # # # #
        for rollout_step in range(args_exp.num_policy_rollouts):

            logging.info(f'Policy rollouts | step {rollout_step}')

            # Reset environment
            # NOTE: this can either be the same env or always a new traffic scene
            next_obs_dict = env.reset()
            controlled_agents = [agent.getID() for agent in env.controlled_vehicles]
            num_agents = len(controlled_agents)
            dict_next_done = {agent_id: False for agent_id in controlled_agents}
            already_done_ids = env.done_ids.copy()
            current_ep_reward = 0

            # Set data buffer
            #TODO: remove hard coding and redo buffer class
            buffer = RolloutBuffer(
                controlled_agents,
                args_exp.num_steps,
                observation_space_dim,
                action_space_dim,
                device,
            )
            
            # Adapt learning rate based on how far we are in the learning process
            if args_exp.anneal_lr: 
                frac = 1.0 - (iter - 1.0) / args_exp.total_iters
                lrnow = frac * args_exp.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            # # # #  Interact with environment  # # # #
            for step in range(0, args_exp.num_steps):
                
                global_step += 1

                for agent_id in controlled_agents:
                    buffer.dones[agent_id][step] = dict_next_done[agent_id] * 1
                    if agent_id not in already_done_ids:
                        buffer.observations[agent_id][step, :] = torch.Tensor(
                            next_obs_dict[agent_id]
                        )
                    else:
                        continue

                # Action logic
                with torch.no_grad():
                    # Nocturne expects a dictionary with actions, we create an item
                    # for every agent that is still active (i.e. not done)
                    action_dict = {
                        agent_id: None
                        for agent_id in controlled_agents
                        if agent_id not in already_done_ids
                    }

                    for agent_id in action_dict.keys():
                        # Take an action
                        action, logprob, _, value = agent.get_action_and_value(
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

                # Update episodal rewards
                current_ep_reward += sum(reward_dict.values())

                # Update done agents
                for agent_id in next_obs_dict.keys():
                    if next_done_dict[agent_id]:
                        dict_next_done[agent_id] = True
                        # Fill dones with ones from step where terminal state was reached
                        buffer.dones[agent_id][step:] = 1

                already_done_ids = [
                    agent_id for agent_id, value in dict_next_done.items() if value
                ]

                # End the game early if all agents are done
                if len(already_done_ids) == num_agents or step == args_exp.num_steps:
                    last_step = step
                    logging.info(f"Terminate episode after {step} steps \n")
                    break
                        
            
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            
            # Store rollout scene experience
            obs_tensor[rollout_step, :, :num_agents, :] = dict_to_tensor(buffer.observations).to(device)
            rew_tensor[rollout_step, :, :num_agents] = dict_to_tensor(buffer.rewards).to(device)
            act_tensor[rollout_step, :, :num_agents] = dict_to_tensor(buffer.actions).to(device)
            done_tensor[rollout_step, :, :num_agents] = dict_to_tensor(buffer.dones).to(device)
            logprob_tensor[rollout_step, :, :num_agents] = dict_to_tensor(buffer.logprobs).to(device)
            value_tensor[rollout_step, :, :num_agents] = dict_to_tensor(buffer.values).to(device)
            
            logging.info(f"Episodic_return: {current_ep_reward}")
            writer.add_scalar("charts/episodic_return", current_ep_reward, global_step)
            writer.add_scalar("charts/episodic_length", step, global_step)


        # # # # Compute advantage estimate via GAE on collected experience # # # #
        # Select the last observation for every policy rollout
        next_obs = obs_tensor[:, last_step, :, :].reshape(-1, observation_space_dim) # (N_steps * N_rollouts, D_obs)
        next_done = done_tensor[:, last_step, :].reshape(-1) # (N_steps * N_rollouts)

        # Flatten over rollout x agent dimension
        dones = done_tensor.reshape((args_exp.num_steps, MAX_AGENTS * args_exp.num_policy_rollouts))
        values = value_tensor.reshape((args_exp.num_steps, MAX_AGENTS * args_exp.num_policy_rollouts))
        rewards = rew_tensor.reshape((args_exp.num_steps, MAX_AGENTS * args_exp.num_policy_rollouts))

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(-1)
            advantages= torch.zeros((args_exp.num_steps, MAX_AGENTS * args_exp.num_policy_rollouts))
            lastgaelam = 0
            for t in reversed(range(args_exp.num_steps)):
                if t == args_exp.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args_exp.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args_exp.gamma * args_exp.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values


        # # # #  Optimization   # # # #
        # Convert the dictionaries to tensors, then flatten over (num_steps x agents)
        b_obs = obs_tensor.reshape(-1, observation_space_dim)
        b_logprobs = logprob_tensor.reshape(-1)
        b_actions = act_tensor.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # Optimizing the policy and value network
        # Since in our multi-agent env some agents finsh earlier then others,
        # there will be entries without observations. We filter these out, as we
        # only want to train on valid sequences
        valid_b_inds = (
            torch.nonzero(torch.any(b_obs != 0, dim=1), as_tuple=False)[:, 0]
        )

        clipfracs = []
        batch_size = len(valid_b_inds)
        minibatch_size = batch_size // num_agents
        for epoch in range(args_exp.update_epochs):
            logging.info(f"Epoch: {epoch}")
            
            # Shuffle batch indices 
            indices = torch.randperm(valid_b_inds.size(0))
            valid_b_inds = valid_b_inds[indices]

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = valid_b_inds[start:end]

                # COMPUTE KL DIV TO HUMAN ANCHOR POLICY
                action, log_prob, tau_dist = human_anchor_policy(b_obs[mb_inds])
                actor_dist = agent.get_policy(b_obs[mb_inds])
                kl_div = kl_loss(tau_dist.probs, actor_dist.probs)

                # Check if the minibatch has at least two elements
                if len(mb_inds) < 2:
                    continue  # Skip this minibatch and move to the next one

                # Compute new logprobs of state dist
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
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
                    - args_exp.lam * kl_div
                )

                logging.info(f"L = {loss}")

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args_exp.max_grad_norm)
                optimizer.step()

            if args_exp.target_kl is not None:
                if approx_kl > args_exp.target_kl:
                    break

        # Clear buffer
        # logging.info("optim step done: clear buffer \n")
        # buffer.clear()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        # TODO: something goes wrong with computing the returns
        # var_y = np.var(y_true)
        # explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/loss", loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar(
            "losses/entropy", entropy_loss.item(), global_step
        )
        writer.add_scalar("charts/mean_advantage", b_advantages.mean())
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        logging.info(f"SPS: {int(global_step / (time.time() - start_time))}")
        logging.info(f"loss: {loss}")
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    env.close()
    writer.close()