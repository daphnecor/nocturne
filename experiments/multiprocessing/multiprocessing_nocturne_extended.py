"""Demo showing multiprocessing with Nocturne is blocked on cluster."""
import torch
from copy import copy, deepcopy
from itertools import repeat
from multiprocessing import Pool
import os
import time

from base_env import BaseEnv
import yaml
import utils
from constants import PPOExperimentConfig
from rl_models import Agent
from nocturne import Action
import torch.multiprocessing as mp

config_path = "/scratch/dc4971/nocturne/experiments/human_regularized/rl_config.yaml"
with open(config_path, "r") as stream:
    rl_config = yaml.safe_load(stream)
args_rl_env = dict(rl_config)


def f(args_exp, Env, args_rl_env, rollout_buffer, ppo_agent):

    start_scene_load = time.perf_counter()
    env = Env(args_rl_env)      
    next_obs_dict = env.reset() 
    end_scene_load = time.perf_counter()
    #print(f'(1/2) Loading scene from process {os.getpid()} took {end_scene_load - start_scene_load:.2f}')


    # Interact with environment
    start_env_steps = time.perf_counter()    
    for step in range(10):

        moving_vehs = env.scenario.getObjectsThatMoved()
        agent_ids = [agent.id for agent in moving_vehs]

        # Select action using policy network 
        with torch.no_grad():
            
            action_dict = {agent_id: None for agent_id in agent_ids}
            
            for agent_id in action_dict.keys():
                action, logprob, _, value = ppo_agent.get_action_and_value(
                    torch.Tensor(next_obs_dict[agent_id])
                )

        # Execute action in env
        next_obs_dict, reward_dict, next_done_dict, info_dict = env.step(
            action_dict
        )

        # next_obs_dict, reward_dict, next_done_dict, info_dict = env.step({
        #     veh.id: Action(acceleration=2.0, steering=1.0, head_angle=0.5)
        #     for veh in moving_vehs
        # })


        # Store info
        agent_idx = 0
        for veh in moving_vehs:
            rollout_buffer.observations[agent_idx][step] = torch.Tensor(next_obs_dict[veh.id])
            agent_idx += 1

    end_env_steps = time.perf_counter()

    return rollout_buffer


if __name__ == '__main__':
    # To use CUDA in subprocesses, one must use either forkserver
    mp.set_start_method('forkserver', force=True)

    print(torch.backends.mps.is_available())

    NUM_TASKS = 16
    NUM_PROCESSES = 8
    
    print(f'{os.cpu_count()} cpus available on cluster. Using {NUM_PROCESSES} processes for this task.') 

    # Prepare arguments  
    args_rl_envs = [deepcopy(args_rl_env) for _ in range(NUM_TASKS)]
    args_exp = PPOExperimentConfig()
    args_exps = [copy(args_exp) for _ in range(NUM_TASKS)]
    tmp_env = BaseEnv(args_rl_env)
    obs_space_dim = tmp_env.observation_space.shape[0]
    act_space_dim = tmp_env.action_space.n

    # Initialize actor and critic models
    ppo_agent = Agent(obs_space_dim, act_space_dim).to("cpu")
    ppo_agent.share_memory() # gradients are allocated lazily, so they are not shared here
    
    # Make rollout buffer
    moving_vehs = tmp_env.scenario.getObjectsThatMoved()
    rollout_buffer = utils.RolloutBufferAdapted(
        len(moving_vehs),
        80,
        tmp_env.observation_space.shape[0],
        tmp_env.action_space.n,
        "cpu",
    )
    
    # SERIAL PROCESSING
    start_time = time.perf_counter()
    for idx in range(NUM_TASKS):
        roll_buffer = f(args_exps[idx], BaseEnv, args_rl_envs[idx], rollout_buffer, ppo_agent)
    end_time = time.perf_counter()
    
    print(f"Processing {NUM_TASKS} nocturne scenes w/o Pool took {end_time - start_time:.2f} s")
    

    # MULTIPROCESSING
    start_time = time.perf_counter()
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        roll_buffer = pool.starmap(f, zip(args_exps, repeat(BaseEnv), args_rl_envs, repeat(deepcopy(rollout_buffer)), repeat(ppo_agent)))#repeat(deepcopy(ppo_agent))))
    
    # ALTERNATIVE APPROACH THAT DOES THE SAME
    # processes = []
    # for rank in range(NUM_PROCESSES):
    #     p = mp.Process(target=f, args=(args_exps, BaseEnv, args_rl_env, rollout_buffer, ppo_agent))
        
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

    end_time = time.perf_counter()
    print(f"Processing {NUM_TASKS} nocturne scenes w/  Pool took {end_time - start_time:.2f} s")
    