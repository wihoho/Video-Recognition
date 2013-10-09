import time
from multiprocessing import Process, Array, Lock
import numpy as np

def func(val, lock, startIndex):
    dis = val.value
    dis[startIndex][0] = 1

if __name__ == '__main__':

    distance = np.zeros((195, 195))

    v = Array('d', distance)
    lock = Lock()
    procs = [Process(target=func, args=(v, lock, i)) for i in range(10)]

    for p in procs: p.start()
    for p in procs: p.join()

    print v.value