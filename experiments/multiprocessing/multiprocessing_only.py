"""Demo showing multiprocessing works on cluster."""

from multiprocessing import Pool
import os
import time


def f():
    print(f"Hello from worker {os.getpid()}")
    time.sleep(2)


if __name__ == '__main__':
    
    NUM_PROCESSES = 2
    NUM_TASKS = 4
    
    print(f"Hello from parent {os.getpid()}")

    start_time = time.perf_counter()
    for x in range(NUM_TASKS):
        _ = f()
    end_time = time.perf_counter()
    print(f"Duration w/o Pool: {end_time - start_time}")
    
    start_time = time.perf_counter()
    with Pool(processes=NUM_PROCESSES) as pool:
        _ = pool.starmap(f, [tuple() for _ in range(NUM_TASKS)])
    
    end_time = time.perf_counter()
    print(f"Duration w/ Pool: {end_time - start_time}")