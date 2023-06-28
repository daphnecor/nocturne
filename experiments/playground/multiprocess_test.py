import datetime
import glob
import logging
import random
import time

from multiprocess import Pool
from multiprocess.dummy import Pool as ThreadPool

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
#from torch.profiler import profile, schedule, tensorboard_trace_handler

import wandb
import yaml

from base_env import BaseEnv
from cfgs.config import set_display_window
from constants import (
    PPOExperimentConfig,
    NocturneConfig,
    WandBSettings,
    HumanPolicyConfig,
)
from dataclasses import asdict, dataclass
from imit_models import BehavioralCloningAgentJoint
from nocturne import Action

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
        ).to(device)
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


    def policy_rollout(env, ppo_agent, obs_dim, action_dim, learning_rate, args_exp):
        """Performs a complete policy rollout.
        
        Args:
            env: Copy of the environment.
            policy: Copy of the policy network.
        Return:

        """

        # Reset environment and get info
        next_obs_dict = env.reset()
        controlled_agents = [agent.getID() for agent in env.controlled_vehicles]
        num_agents = len(controlled_agents)
        dict_next_done = {agent_id: False for agent_id in controlled_agents}
        already_done_ids = env.done_ids.copy()
        current_ep_reward = 0

        # Set data buffer for within scene logging
        buffer = RolloutBuffer(
            controlled_agents,
            args_exp.num_steps,
            obs_dim,
            action_dim,
            device,
        )

        # # # #  Interact with environment  # # # #
        for step in range(args_exp.num_steps):

            for agent_id in controlled_agents:
                buffer.dones[agent_id][step] = dict_next_done[agent_id] * 1
                if agent_id not in already_done_ids:
                    buffer.observations[agent_id][step, :] = torch.Tensor(
                        next_obs_dict[agent_id]
                    )
                else:
                    continue
            
            # SELECT ACTIONS
            with torch.no_grad():

                action_dict = {
                    agent_id: None
                    for agent_id in controlled_agents
                    if agent_id not in already_done_ids
                }

                for agent_id in action_dict.keys():
                    
                    # Take an action
                    action, logprob, _, value = ppo_agent.get_action_and_value(
                        torch.Tensor(next_obs_dict[agent_id]).to(device)
                    )
                    buffer.values[agent_id][step] = value.flatten()
                    buffer.actions[agent_id][step] = action
                    buffer.logprobs[agent_id][step] = logprob
                    action_dict[agent_id] = action.item()

                # TAKE A STEP IN ENV
                next_obs_dict, reward_dict, next_done_dict, info_dict = env.step(
                    action_dict
                )

                # Store rewards
                for agent_id in next_obs_dict.keys():
                    buffer.rewards[agent_id][step] = torch.from_numpy(
                        np.asarray(reward_dict[agent_id])
                    )

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

                # ONLY END GAME EARLY WHEN ALL AGENTS ARE DONE
                if len(already_done_ids) == num_agents:
                    last_step = step
                    break
                
            return (
                buffer.observations,
                buffer.actions,
                buffer.values,
                buffer.logprobs,
                buffer.dones,
            )


                

if __name__ == "__main__":

    # Configs
    args_exp = PPOExperimentConfig()
    args_wandb = WandBSettings()
    args_env = NocturneConfig()
    args_human_pol = HumanPolicyConfig()

    # Logging
    now = datetime.datetime.now()
    formatted_time = now.strftime("%D%H%M")
    run_name = f"{args_env.env_id}__{args_wandb.exp_name}_{formatted_time}"
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args_exp).items()])),
    )

    if args_wandb.track:
        wandb.init(
            project=args_wandb.project_name,
            #sync_tensorboard=True,
            config=asdict(args_wandb),
            group="nocturne",
            name=run_name,
            save_code=True,
        )

    # Seeding
    random.seed(args_exp.seed)
    np.random.seed(args_exp.seed)
    torch.manual_seed(args_exp.seed)
    torch.backends.cudnn.deterministic = args_exp.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args_exp.cuda else "cpu"
    )

    logging.critical(f'DEVICE: {device}')

    # Env setup
    set_display_window()
    env = make_env(args_env.nocturne_rl_cfg, args_exp.seed, run_name)


    # Do policy rollouts
    # Storage
    obs_tensor = torch.zeros(
        (
            args_exp.num_policy_rollouts,
            args_exp.num_steps,
            MAX_AGENTS,
            observation_space_dim,
        )
    ).to(device)
    rew_tensor = torch.zeros(
        (args_exp.num_policy_rollouts, args_exp.num_steps, MAX_AGENTS)
    ).to(device)
    act_tensor = torch.zeros(
        (args_exp.num_policy_rollouts, args_exp.num_steps, MAX_AGENTS)
    ).to(device)
    done_tensor = torch.zeros(
        (args_exp.num_policy_rollouts, args_exp.num_steps, MAX_AGENTS)
    ).to(device)
    logprob_tensor = torch.zeros(
        (args_exp.num_policy_rollouts, args_exp.num_steps, MAX_AGENTS)
    ).to(device)
    value_tensor = torch.zeros(
        (args_exp.num_policy_rollouts, args_exp.num_steps, MAX_AGENTS)
    ).to(device)
    veh_coll_tensor = torch.zeros((args_exp.num_policy_rollouts)).to(device)
    edge_coll_tensor = torch.zeros((args_exp.num_policy_rollouts)).to(device)
    goal_tensor = torch.zeros((args_exp.num_policy_rollouts)).to(device)
    
    # RUN: CONT HERE.
    #policy_rollout(env, ppo_agent, obs_dim, action_dim, learning_rate, args_exp)


