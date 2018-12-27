# cython: language_level=3

import numpy as np

cimport numpy as np

cimport cython

from cython.parallel import prange

DTYPE = np.int

ctypedef np.int_t DTYPE_t

cdef int DIR[8][2]

DIR = [
[-1, 0],
[-1, 1],
[ 0, 1],
[ 1, 1],
[ 1, 0],
[1, -1],
[0, -1],
[-1,-1],]

@cython.boundscheck(False)
@cython.wraparound(False)
def place(np.ndarray[DTYPE_t, ndim=2] board, int x, int y, int color):
    if x < 0:
        return False
    board[x][y] = color
    cdef bint valid = False
    cdef int i, j, d
    for d in range(8):
        i = x + DIR[d][0]
        j = y + DIR[d][1]
        while 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == -color:
            i += DIR[d][0]
            j += DIR[d][1]
        if 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == color:
            while True:
                i -= DIR[d][0]
                j -= DIR[d][1]
                if i == x and j == y:
                    break
                valid = True
                board[i][j] = color
    return valid

cdef bint cplace(int board[8][8], int x, int y, int color) nogil:
    if x < 0:
        return False
    board[x][y] = color
    cdef bint valid = False
    cdef int i, j, d
    for d in range(8):
        i = x + DIR[d][0]
        j = y + DIR[d][1]
        while 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == -color:
            i += DIR[d][0]
            j += DIR[d][1]
        if 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == color:
            while True:
                i -= DIR[d][0]
                j -= DIR[d][1]
                if i == x and j == y:
                    break
                valid = True
                board[i][j] = color
    return valid

def get_actions(np.ndarray[DTYPE_t, ndim=2] board, int color):
    cdef int cboard[8][8]
    cdef int new_cboard[8][8]
    cdef int i, j, id, ii, jj, k
    for ii in range(8):
        for jj in range(8):
            cboard[ii][jj] = board[ii][jj]
    cdef int moves[64]
    for k in range(64):
        moves[k] = 0
    for id in prange(64, nogil=True):
        i = int(id / 8)
        j = id % 8
        if cboard[i][j] == 0:
            for ii in range(8):
                for jj in range(8):
                    new_cboard[ii][jj] = cboard[ii][jj]
            if cplace(new_cboard, i, j, color):
                moves[id] = 1
    ret = []
    for id in range(64):
        i = int(id / 8)
        j = id % 8
        if moves[id] == 1:
            ret.append((i,j))
    return ret
