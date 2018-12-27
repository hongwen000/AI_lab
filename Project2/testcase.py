import numpy as np
import time

values = np.array([[ 30, -25, 10, 5, 5, 10, -25,  30,],
    [-25, -25,  1, 1, 1,  1, -25, -25,],
    [ 10,   1,  5, 2, 2,  5,   1,  10,],
    [  5,   1,  2, 1, 1,  2,   1,   5,],
    [  5,   1,  2, 1, 1,  2,   1,   5,],
    [ 10,   1,  5, 2, 2,  5,   1,  10,],
    [-25, -25,  1, 1, 1,  1, -25, -25,],
    [ 30, -25, 10, 5, 5, 10, -25,  30,]])

grid = np.array([[ 1, 1, 1, 1, 0, 0, 0, 0],
    [ 1,-1, 1, 1, 1,-1, 0, 0],
    [ 1,-1, 1, 1, 1,-1,-1,-1],
    [ 1,-1, 1, 1, 1,-1,-1,-1],
    [ 1,-1,-1, 1, 1, 1,-1,-1],
    [ 1,-1,-1, 1, 1, 1,-1,-1],
    [ 1, 0, 1, 1, 1, 1, 0, 0],
    [ 1, 1, 1, 1, 1, 1, 0, 0]])

ret = 0
s = time.monotonic_ns()
for i in range(1000000):
    # ret = np.tensordot(values, grid, axes=((0,1),(0,1)))
    tmp = values * grid;
    ret = np.sum(tmp)
e = time.monotonic_ns()
print(e - s)
print(ret)
