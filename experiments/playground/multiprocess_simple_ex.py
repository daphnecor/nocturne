import os
import time
import multiprocessing
from multiprocessing import Process, Pool
import numpy as np

def perform_policy_rollout(env):
    """Do a policy rollout"""
    start_t = time.perf_counter()
    

    obs = np.random.rand(10_000, 10_000)

    end_t = time.perf_counter()

    return obs, end_t - start_t 

def collect_experience(envs, processes=None, chunksize=None):
    """Store data from K policy rollouts"""

    start_t = time.perf_counter()
    
    all_obs = []

    print("starting rollouts")
    with Pool() as pool:
        results = pool.map(perform_policy_rollout, envs)

        for obs, duration, in results:
            all_obs.append(obs)

            print(f"{obs.shape} completed in {duration:.2f}")

    end_t = time.perf_counter()
    total_duration = end_t - start_t
    print(f"Rollouts took {total_duration:.2f}s total")
    return all_obs


if __name__ == "__main__":

    envs = [1, 2, 3, 4, 5, 5, 5]
    
    # Determine the chunk size: split items into chunks, so that each worker
    # i.e. one process in the pool, grabs an entire chunk of work. Having large
    # chunk sizes is faster but uses more memory, small ones
    chunksize = 2

    print(f'Number of available cpus: {multiprocessing.cpu_count()}')

    all_obs = collect_experience(envs)
    
    start_t = time.perf_counter()
    obs = np.random.rand(10_000, 10_000)
    end_t = time.perf_counter()
    print(f'single arr takes: {end_t - start_t } s')