from multiprocessing import Pool, TimeoutError
import numpy as np
import time
import os

def do_policy_rollout():
    process_id = os.getpid()
    start_time = time.perf_counter()
    # Your code for policy rollout here
    rollout_data = np.random.rand(1000000)  # Example rollout data
    end_time = time.perf_counter()
    duration = end_time - start_time
    return rollout_data, duration, process_id

def sequential_policy_rollouts(num_rollouts):
    rollout_results = []
    for _ in range(num_rollouts):
        rollout_results.append(do_policy_rollout())
    return rollout_results

if __name__ == '__main__':
    num_rollouts = 1000

    # Parallel execution using multiprocessing.Pool
    with Pool() as pool:
        parallel_start_time = time.perf_counter()
        parallel_rollout_results = [pool.apply_async(do_policy_rollout) for _ in range(num_rollouts)]
        parallel_results = [res.get() for res in parallel_rollout_results]
        parallel_end_time = time.perf_counter()
        parallel_duration = parallel_end_time - parallel_start_time

    # Sequential execution without multiprocessing
    sequential_start_time = time.perf_counter()
    sequential_results = sequential_policy_rollouts(num_rollouts)
    sequential_end_time = time.perf_counter()
    sequential_duration = sequential_end_time - sequential_start_time

    # Compare durations
    print("Parallel Duration:", parallel_duration, "seconds")
    print("Sequential Duration:", sequential_duration, "seconds")

    # Check if parallel execution is faster
    if parallel_duration < sequential_duration:
        print("Parallel execution is faster.")
    else:
        print("Sequential execution is faster.")

    # Print process IDs for each rollout in parallel execution
    print("Process IDs:")
    for _, _, process_id in parallel_results:
        print(process_id)
