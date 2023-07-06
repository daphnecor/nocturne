from copy import deepcopy
from itertools import repeat
from multiprocessing import Pool
import os
import time
from nocturne import Action

from base_env import BaseEnv
from rl_models import Agent
import torch
import utils
import numpy as np

my_value = None
    
def f(x, Env, args_rl_env, agent):
    time.sleep(2)
    env = Env(args_rl_env)
    next_obs_dict = env.reset()

    moving_vehs = env.scenario.getObjectsThatMoved()

    # Set data buffer for within scene logging
    rollout_buffer = utils.RolloutBuffer(
        [agent.id for agent in moving_vehs],
        80,
        env.observation_space.shape[0],
        env.action_space.n,
        "cpu", # args_exp.device,
    )
    
    for timestep in range(80):

        # # # #  Interact with environment  # # # #
        next_obs_dict, reward_dict, next_done_dict, info_dict = env.step({
            veh.id: Action(acceleration=2.0, steering=1.0, head_angle=0.5)
            for veh in moving_vehs
        })

        with torch.no_grad():
            action = agent.get_action_and_value(torch.zeros(size=(5, )))

        # Update tensors
        for veh in moving_vehs:
            agent_id = veh.id
            rollout_buffer.rewards[agent_id][timestep] = torch.ones(size=(1,))


        

    return rollout_buffer


if __name__ == '__main__':
    
    NUM_TASKS = 2
    NUM_PROCESSES = 2
    OBS_SPACE_DIM = 5
    ACT_SPACE_DIM = 5

    RL_SETTINGS_PATH = "experiments/human_regularized/rl_config.yaml"
    args_rl_env = utils.load_yaml_file(RL_SETTINGS_PATH)

    ppo_agent = Agent(OBS_SPACE_DIM, ACT_SPACE_DIM)

    start_time = time.time()
    for x in range(NUM_TASKS):
        rollout_buffers = f(x, BaseEnv, args_rl_env, deepcopy(ppo_agent))
    end_time = time.time()
    print(f"Duration w/o Pool: {end_time - start_time}")
    
    with Pool(processes=NUM_PROCESSES) as pool:
        start_time = time.time()
        rollout_buffers = pool.starmap(f, zip(range(NUM_TASKS), repeat(BaseEnv), repeat(args_rl_env), repeat(deepcopy(ppo_agent))))
        end_time = time.time()
        print(f"Duration w/ Pool: {end_time - start_time}")
