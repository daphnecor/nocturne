"""Demo showing multiprocessing works on cluster computing a sum."""

import time
from multiprocessing import Pool
import os

def sum_square(number):
    s = 0
    for i in range(number):
        s += i * i
    return s


def sum_square_with_mp(numbers):

    start_time = time.time()
    p = Pool(processes=NUM_PROCESSES)
    result = p.map(sum_square, numbers)

    p.close()
    p.join()

    print(f"Processing {len(numbers)} numbers w/ Pool took: {time.time() - start_time:.2f} s")


def sum_square_no_mp(numbers):

    start_time = time.time()
    result = []

    for i in numbers:
        result.append(sum_square(i))

    print(f"Processing {len(numbers)} numbers w/o Pool took: {time.time() - start_time:.2f} s")


if __name__ == '__main__':

    NUM_PROCESSES = 2
    print(f'{os.cpu_count()} cpus available on cluster. Using: {NUM_PROCESSES} for this task.')
    numbers = range(20_000)
    
    sum_square_no_mp(numbers)
    sum_square_with_mp(numbers)