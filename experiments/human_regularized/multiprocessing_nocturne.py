"""Demo showing multiprocessing with Nocturne is blocked on cluster."""

from copy import deepcopy
from itertools import repeat
from multiprocessing import Pool
import os
import time

from base_env import BaseEnv
import yaml
import utils

config_path = "/scratch/dc4971/nocturne/experiments/human_regularized/rl_config.yaml"
with open(config_path, "r") as stream:
    rl_config = yaml.safe_load(stream)
args_rl_env = dict(rl_config)


def f(Env, args_rl_env):
    env = Env(args_rl_env)       # Blocks: Creating Simulation instance is blocking, unrelated to input file
    next_obs_dict = env.reset()  # Blocks: Creating Simulation instance is blocking, unrelated to input file


if __name__ == '__main__':
    
    NUM_TASKS = 4
    NUM_PROCESSES = 2
    
    args_rl_envs = [deepcopy(args_rl_env) for _ in range(NUM_TASKS)]
    
    print(f"Hello from parent {os.getpid()}")

    start_time = time.perf_counter()
    for idx in range(NUM_TASKS):
        _ = f(BaseEnv, args_rl_envs[idx])
    end_time = time.perf_counter()
    print(f"Duration w/o Pool: {end_time - start_time}")
    
    start_time = time.perf_counter()
    with Pool(processes=NUM_PROCESSES) as pool:
        rollout_buffers = pool.starmap(f, zip(repeat(BaseEnv), args_rl_envs))
    
    end_time = time.perf_counter()
    print(f"Duration w/ Pool: {end_time - start_time}")