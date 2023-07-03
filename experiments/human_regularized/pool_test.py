from copy import deepcopy
from itertools import repeat
from multiprocessing import Pool
import os
import time

from base_env import BaseEnv
from rl_models import Agent
import torch
import utils

my_value = None
    
def f(x, Env, args_rl_env, agent):
    time.sleep(2)
    env = Env(args_rl_env)
    env.reset()

    agent.get_action_and_value(torch.zeros(size=(5, )))

    for step in range(10):
        env.step()

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
        f(x, BaseEnv, args_rl_env, deepcopy(ppo_agent))
    end_time = time.time()
    print(f"Duration w/o Pool: {end_time - start_time}")
    
    with Pool(processes=NUM_PROCESSES) as pool:
        start_time = time.time()
        pool.starmap(f, zip(range(NUM_TASKS), repeat(BaseEnv), repeat(args_rl_env), repeat(deepcopy(ppo_agent))))
        end_time = time.time()
        print(f"Duration w/ Pool: {end_time - start_time}")