import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import wandb
#import pdb

import yaml
from base_env import BaseEnv
import logging
from imit_models import BehavioralCloningAgentJoint

logging.basicConfig(level=logging.INFO)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl_env_cfg", type=str, default="experiments/human_regularized/rl_config.yaml",
        help="settings for the nocturne environment.")
    parser.add_argument("--lam", type=float, default=0,
        help="coefficient of kl_div to human anchor policy")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="human_regularized_rl",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Nocturne-v0",
        help="the id of the environment")
    parser.add_argument("--total-iters", type=int, default=4, #1000, #50_000,
        help="total iterations of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-steps", type=int, default=80,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    # fmt: on
    return args


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
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            group="nocturne",
            name=run_name,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Env setup
    env = make_env(args.rl_env_cfg, args.seed, run_name)

    # State and action space dimension
    observation_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.n

    # Create ppo agent
    agent = Agent(env, observation_space_dim, action_space_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    # LOAD HUMAN ANCHOR POLICY
    human_policy_cfg = {
        "batch_size": 1,
        "hidden_layers": [1025, 256, 128],
        "actions_discretizations": [
            15,
            42,
        ],
        "actions_bounds": [
            [-6, 6],
            [-0.7, 0.7],
        ],
    }
    human_anchor_policy = BehavioralCloningAgentJoint(
        num_inputs=observation_space_dim,
        config=human_policy_cfg,
        device=device,
    ).to(device)
    human_anchor_policy.load_state_dict(
        torch.load("experiments/human_regularized/human_anchor_policy_AS.pth")
    )

    # Setup
    global_step = 0 
    start_time = time.time()
    num_updates = args.total_iters

    for update in range(1, num_updates + 1):  # For a number of iterations
        # We have a multi-agent setup, where the number of agents varies per traffic scene
        # So, we create a buffer to store all agent trajectories
        next_obs_dict = env.reset()
        controlled_agents = [agent.getID() for agent in env.controlled_vehicles]
        num_agents = len(controlled_agents)
        buffer = RolloutBuffer(
            controlled_agents,
            args.num_steps,
            observation_space_dim,
            action_space_dim,
            device,
        )
        
        current_ep_reward = 0
        dict_next_done = {agent_id: False for agent_id in controlled_agents}
        already_done_ids = env.done_ids.copy()

        logging.info(f"--- iter: {update} --- \n")
        logging.info(f"Initializing new scene with {num_agents} agents.")
        logging.info(f"Done ids = {already_done_ids}")
        writer.add_scalar("charts/num_agents_in_scene", num_agents, global_step)

        # Annealing the rate if instructed to do so
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Rollout phase
        logging.info("--- POLICY ROLLOUTS --- \n")

        for step in range(0, args.num_steps):
            logging.debug(f"Step: {step}")

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

            # Agents take actions simultaneously
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

            logging.debug(f"step reward: {sum(reward_dict.values()):.2f}")
            logging.debug(f"cumsum reward: {current_ep_reward:.2f}")

            # writer.add_scalar(
            #     "charts/cum_step_return_overall", current_ep_reward, global_step
            # )
            # writer.add_scalar(
            #     "charts/cum_step_return_norm",
            #     current_ep_reward / num_agents,
            #     global_step,
            # )

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
            if len(already_done_ids) == num_agents:
                logging.info(f"Terminate episode after {step} steps \n")
                break

        # Compute goal achieved, collision rate etc
        total_goal_achieved = sum(1 for item in info_dict.values() if item['goal_achieved'])
        total_collided = sum(1 for item in info_dict.values() if item['collided'])
        total_veh_veh_collided = sum(1 for item in info_dict.values() if item['veh_veh_collision'])
        total_veh_edge_collided = sum(1 for item in info_dict.values() if item['veh_edge_collision'])

        logging.info(f"episodic_return: {current_ep_reward}")
        writer.add_scalar("charts/episodic_return", current_ep_reward, global_step)
        writer.add_scalar("charts/episodic_length", step, global_step)
        writer.add_scalar("charts/goal_achieved", total_goal_achieved, update)
        writer.add_scalar("charts/agents_collided", total_collided, update)
        writer.add_scalar("charts/veh_veh_collisions", total_veh_veh_collided, update)
        writer.add_scalar("charts/veh_edge_collisions", total_veh_edge_collided, update)

        # Bootstrap value if not done
        logging.info(f"--- BOOTSTRAP ---")

        # Take last obseration (num_agents x obs_dim)
        next_obs = dict_to_tensor(buffer.observations)[step].reshape(
            (num_agents, observation_space_dim)
        )
        next_done = torch.Tensor(list(dict_next_done.values())).to(device) # Most recent done dict
        dones = dict_to_tensor(buffer.dones).to(device)
        values = dict_to_tensor(buffer.values).to(device)
        rewards = dict_to_tensor(buffer.rewards).to(device)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros((args.num_steps, num_agents)).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        logging.info("--- OPTIMIZE POLICY NETWORK --- \n")

        # Learning phase
        # Convert the dictionaries to tensors, then flatten over (num_steps x agents)
        b_obs = dict_to_tensor(buffer.observations).reshape(
            -1, observation_space_dim
        )  # (batch_size x obs_space_dim)
        b_logprobs = dict_to_tensor(buffer.logprobs).reshape(-1) # (batch_size x 1)
        b_actions = dict_to_tensor(buffer.actions).reshape(-1)
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
        for epoch in range(args.update_epochs):
            logging.info(f"Epoch: {epoch}")
            np.random.shuffle(valid_b_inds)
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

                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + v_loss * args.vf_coef
                    - args.lam * kl_div
                )

                # logging.info(f'policy loss: {pg_loss:.3f} | val loss: {v_loss * args.vf_coef:.3f}\n')
                logging.info(f"L = {loss}")

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
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
            "losses/weighted_entropy", args.ent_coef * entropy_loss.item(), global_step
        )
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