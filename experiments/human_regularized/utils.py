from typing import List
import yaml
from multiprocessing import Manager
import numpy as np
import torch
from nocturne import Simulation
import nocturne_gym as gym

def do_policy_rollout(args, env, ppo_agent, device):
    """
    Gather new trajectories (rollouts) with the current policy.
    Args:
        args: Experiment configurations.
        env: Nocturne RL environment.
        ppo_agent: Most recent actor critic network.
        device: torch.device.
    Return:
        buffer
    """

    # Reset environment 
    next_obs_dict = env.reset()
    obs_space_dim = env.observation_space.shape[0]
    act_space_dim = env.action_space.n

    controlled_agents = [agent.getID() for agent in env.controlled_vehicles]
    num_agents = len(controlled_agents)

    # Storage     
    dict_next_done = {agent_id: False for agent_id in controlled_agents}
    already_done_ids = env.done_ids.copy()
    current_ep_rew_agents = {agent_id: 0 for agent_id in controlled_agents}

    # Set data buffer for within scene logging
    buffer = RolloutBuffer(
        controlled_agents,
        args.num_steps,
        obs_space_dim,
        act_space_dim,
        device,
    )
    veh_veh_collisions = 0
    veh_edge_collisions = 0
    veh_goal_achieved = 0

    # # # #  Interact with environment  # # # #
    for step in range(args.num_steps):

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

        # Check for collisions and whether goal is achieved
        veh_veh_collisions += sum(
            info["veh_veh_collision"] for info in info_dict.values()
        )
        veh_edge_collisions += sum(
            info["veh_edge_collision"] for info in info_dict.values()
        )
        veh_goal_achieved += sum(
            info["goal_achieved"] for info in info_dict.values()
        )

        # Update done agents
        for agent_id in next_obs_dict.keys():
            current_ep_rew_agents[agent_id] += reward_dict[agent_id].item()
            if next_done_dict[agent_id]:
                dict_next_done[agent_id] = True

        already_done_ids = [
            agent_id for agent_id, value in dict_next_done.items() if value
        ]

        # End the game early if all agents are done
        if len(already_done_ids) == num_agents or step == args.num_steps:
            break
    
    return buffer, veh_veh_collisions, veh_edge_collisions, veh_goal_achieved, num_agents



def render_scene(env, render_mode, window_size, ego_vehicle=None, view_dist=None, view_angle=None, draw_target=True, padding=10.0):
    """
    Renders a nocturne scene.

    Args:
        env (object): The environment object.
        render_mode (str): The rendering mode ("whole_scene" or "agent_view").
        window_size (int): The size of the rendering window.
        ego_vehicle (object, optional): The ego vehicle object. Defaults to None.
        view_dist (float, optional): The viewing distance. Defaults to None.
        view_angle (float, optional): The viewing angle. Defaults to None.
        draw_target (bool, optional): Flag to indicate whether to draw the target. Defaults to True.
        padding (float, optional): The padding value. Defaults to 10.0.

    Returns:
        torch.Tensor: The rendered scene.
    """

    if render_mode == "whole_scene":
        render_scene = env.scenario.getImage(
            img_width=1200,
            img_height=1200,
            padding=10.0,
            draw_target_positions=True,
        )
        
    elif render_mode == "agent_view":
        render_scene = env.scenario.getConeImage(
            source=ego_vehicle,
            view_dist=view_dist,
            view_angle=view_angle,
            head_angle=0,
            img_width=window_size,
            img_height=window_size,
            padding=padding,
            draw_target_position=draw_target,
        )
         
    return render_scene.T


def find_last_zero_index(tensor):
    """
    Finds the index of the last occurrence of zero in each row of a tensor.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The indices of the last occurrence of zero in each row.
    """

    num_items, num_steps = tensor.shape
    last_zero_indices = torch.zeros(num_items, dtype=torch.long)
    
    for i in range(num_items):
        found_zero = False
        for j in range(num_steps-1, -1, -1):
            if tensor[i, j] == 0:
                last_zero_indices[i] = j
                found_zero = True
                break
        if not found_zero:
            last_zero_indices[i] = -1
    
    return last_zero_indices


def dict_to_tensor(my_dict):
    """
    Converts a dictionary of tensors to a stacked tensor.

    Args:
        my_dict (dict): The input dictionary.

    Returns:
        torch.Tensor: The stacked tensor.
    """

    tensor_list = []
    for agent_id, tensor in my_dict.items():
        tensor_list.append(torch.Tensor(tensor))
    stacked_tensor = torch.stack(tensor_list, dim=1)
    return stacked_tensor.squeeze()


class RolloutBufferAdapted:
    """
    Shared rollout buffer to store collected trajectories for every agent we control.
    Must be reset after the policy network is updated.
    """
    def __init__(
        self, num_agents, num_steps, obs_space_dim, act_space_dim, device
    ):
        self.observations = self.create_tensor_dict(
            num_agents, num_steps, device, obs_space_dim
        )
        self.actions = self.create_tensor_dict(
            num_agents,
            num_steps,
            device,
        )
        self.logprobs = self.create_tensor_dict(num_agents, num_steps, device)
        self.rewards = self.create_tensor_dict(num_agents, num_steps, device)
        self.dones = self.create_tensor_dict(num_agents, num_steps, device)
        self.values = self.create_tensor_dict(num_agents, num_steps, device)

    def create_tensor_dict(self, num_agents, num_steps, device="cpu", dim=None):
        tensor_dict = {}
        for agent in range(num_agents):
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


def load_yaml_file(config_path):
    try:
        with open(config_path, "r") as stream:
            rl_config = yaml.safe_load(stream)
    except FileNotFoundError:
        print(f"File not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error while parsing YAML file: {e}")
        return None
    return dict(rl_config)
