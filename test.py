


import numpy as np
# import multiprocessing as mp
from multiprocessing import Pool
import timeit

def long_run(n):
    s = ''
    for i in range(int(n)):
        s += str(i)
    return s


def run_parallel():
    pool = Pool(processes=6)
    pools = [pool.apply_async(long_run, args=(x,)) for x in [5e5, 1e5, 7e5, 3e4, 2e4, 1e4, 2e5]]
    results = [pool.get() for pool in pools]
    return results

def run_serial():
    results = [long_run(x) for x in [5e5, 1e5, 7e5, 3e4, 2e4, 1e4, 2e5]]
    return results

if __name__ == '__main__':

    # res = run_parallel()
    # res = run_serial()

    ret = timeit.timeit(lambda: run_serial(), number=10) / 10
    print(f"serial: {ret}")

    ret = timeit.timeit(lambda: run_parallel(), number=10) / 10
    print(f"parallel: {ret}")


