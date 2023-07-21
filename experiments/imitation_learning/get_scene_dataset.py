import numpy as np
import pandas as pd
import sys
import yaml
import torch
import utils
import logging

logging.basicConfig(level=logging.INFO)

from nocturne import Simulation
#TODO: structure files into packagaes
sys.path.insert(1, '/scratch/dc4971/nocturne/experiments/human_regularized/')

from base_env import BaseEnv

def map_actions_to_grid_idx(
    accel_lb=-2, 
    accel_ub=2,
    accel_disc=5,
    steering_lb=-0.7,
    steering_ub=0.7,
    steering_disc=5,
):
    # Make mapping action idx to values 
    accel_grid = np.linspace(
        accel_lb, accel_ub, accel_disc,
    )
    steer_grid = np.linspace(
        steering_lb, steering_ub, steering_disc,
    )

    # Create joint action space
    actions_to_joint_idx = {}
    i = 0
    for accel in accel_grid:
        for steer in steer_grid:
            actions_to_joint_idx[accel, steer] = [i]
            i += 1   

    return accel_grid, steer_grid, actions_to_joint_idx


def get_expert_grid_actions(
        scenario_path, 
        scenario_config,
        tmin=0,
        tmax=90,
        view_dist = 80,
        view_angle = 3.14,
        dt=0.1,
    ):
    """
    Rollout a scene to obtain the expert grid actions for every timestep.
    """
    
    # Create simulation from given scenario
    sim = Simulation(str(scenario_path), scenario_config)
    scenario = sim.getScenario()

    accel_grid, steer_grid, actions_to_joint_idx = map_actions_to_grid_idx()

    # Set objects to be expert-controlled
    for obj in scenario.getObjects():
        obj.expert_control = True

    objects_that_moved = scenario.getObjectsThatMoved()
    objects_of_interest = [
        obj for obj in scenario.getVehicles() if obj in objects_that_moved
    ]

    logging.info(f'objects_of_interest: {[obj.id for obj in objects_of_interest]}')

    # Setup dataframe to store actions
    actions_dict = {}
    for agent in objects_of_interest:
        actions_dict[agent.id] = np.zeros(tmax)
    df_actions = pd.DataFrame(actions_dict)

    for time in range(tmin, tmax):
        for obj in objects_of_interest:
            
            logging.info(f'VEHICLE #{obj.getID()} \n ----')
            
            # get state
            ego_state = scenario.ego_state(obj)
            visible_state = scenario.flattened_visible_state(
                obj, view_dist=view_dist, view_angle=view_angle)
            state = np.concatenate((ego_state, visible_state))

            logging.debug(f'obs: {state}')

            # get expert action
            expert_action = scenario.expert_action(obj, time)
            # check for invalid action (because no value available for taking derivative)
            # or because the vehicle is at an invalid state
            if expert_action is None:
                continue
            
            expert_action = expert_action.numpy()

            acceleration = expert_action[0]
            steering = expert_action[1]

            # Map actions to nearest grid indices
            accel_grid_idx = utils.find_closest_index(accel_grid, acceleration)
            steering_grid_idx = utils.find_closest_index(steer_grid, steering)
            
            # Map to joint action space
            expert_action_idx = actions_to_joint_idx[accel_grid[accel_grid_idx], steer_grid[steering_grid_idx]]

            #print(f'EXPERT action joint index: {expert_action_idx[0]} | original: [ acc = {acceleration:.2f}, steer = {steering:.2f} ]')

            df_actions.loc[time][obj.getID()] = expert_action_idx[0]

        # Step the simulation
        sim.step(dt)

    return df_actions
    
def get_expert_grid_observations(
    base_env_config,
    df_expert_actions,
    tmin=0,
    tmax=90,

):
    """Step through scene with given grid expert actions"""

    # Make environment
    env = BaseEnv(base_env_config)
    next_obs_dict = env.reset()

    # Storage
    expert_action_tensor = torch.zeros((df_expert_actions.shape))
    obs_tensor = torch.zeros((df_expert_actions.shape[0], df_expert_actions.shape[1], env.observation_space.shape[0]))

    # Select agents of interest
    agents_of_interest = [
        agent
        for agent in env.scenario.getVehicles()
        if agent in env.scenario.getObjectsThatMoved()
    ]

    already_done_ids = []

    joint_action_idx_to_values = env.idx_to_actions

    for timestep in range(tmin, tmax):
        
        # Select action from expert grid actions dataframe
        action_dict = {}
        for agent in agents_of_interest:
            if agent.id in next_obs_dict:
                action = int(df_expert_actions[agent.id].loc[timestep])
                action_dict[agent.id] = action

        # Store actions + obervations of living agents
        agent_idx = 0
        for agent_id in next_obs_dict:
            expert_action_tensor[timestep, agent_idx] = action_dict[agent_id]
            obs_tensor[timestep, agent_idx, :] = torch.Tensor(next_obs_dict[agent_id])
            agent_idx += 1

        # Execute actions
        next_obs_dict, reward_dict, next_done_dict, info_dict = env.step(
            action_dict
        )

    # Flatten along the agent dimension
    expert_action_tensor = expert_action_tensor.flatten()
    obs_tensor = obs_tensor.flatten(start_dim=0, end_dim=1)

    # Get the indices where 'expert_action_tensor' is non-zero
    non_zero_indices = torch.nonzero(expert_action_tensor)

    # Select elements from 'other_tensor' based on the non-zero indices along the zero-th axis
    valid_obs_tensors = obs_tensor[non_zero_indices[:, 0]]
    valid_expert_actions = expert_action_tensor[non_zero_indices].squeeze(dim=1)

    return valid_obs_tensors, valid_expert_actions


if __name__ == "__main__":
    
    SCENARIO_PATH = "/scratch/dc4971/nocturne/data/formatted_json_v2_no_tl_train/example_scenario.json"
    PATH_TO_SCENE_CONFIG = "/scratch/dc4971/nocturne/experiments/human_regularized/rl_config.yaml"

    with open(PATH_TO_SCENE_CONFIG, "r") as stream:
        base_env_config = yaml.safe_load(stream)

    scenario_config = {
        'start_time': 0, 
        'allow_non_vehicles': False, 
        'moving_threshold': 0.2, 
        'speed_threshold': 0.05, 
        'max_visible_road_points': 500, 
        'sample_every_n': 1, 
        'road_edge_first': False
    }

    # Get expert actions mapped to nearest grid indices
    df_expert_actions = get_expert_grid_actions(
        SCENARIO_PATH,
        scenario_config,
    )

    # Now step through the environment and obtain the correct observations for every action 
    obs_tensor, expert_action_tensor = get_expert_grid_observations(
        base_env_config,
        df_expert_actions,
    )

    # Save data
    torch.save(expert_action_tensor, f"experiments/imitation_learning/data_expert_grid/expert_actions.pt")
    torch.save(obs_tensor, f"experiments/imitation_learning/data_expert_grid/observations.pt")
