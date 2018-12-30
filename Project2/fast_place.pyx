# cython: language_level=3
#cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np

import random

cimport numpy as np

cimport cython

from cython.parallel import prange

DTYPE = np.int

ctypedef np.int_t DTYPE_t

cdef int dir[8][2]

dir = [
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
def place(np.ndarray[DTYPE_t, ndim=2] board, int x, int y, int color) -> object:
    if x < 0:
        return False
    board[x][y] = color
    cdef bint valid = 0
    cdef int i, j, d
    for d in range(8):
        i = x + dir[d][0]
        j = y + dir[d][1]
        while 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == -color:
            i += dir[d][0]
            j += dir[d][1]
        if 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == color:
            while True:
                i -= dir[d][0]
                j -= dir[d][1]
                if i == x and j == y:
                    break
                valid = 1
                board[i][j] = color
    return valid





cdef bint cplace(int board[8][8], int x, int y, int color) nogil:
    if x < 0:
        return False
    board[x][y] = color
    cdef bint valid = False
    cdef int i, j, d
    for d in range(8):
        i = x + dir[d][0]
        j = y + dir[d][1]
        while 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == -color:
            i += dir[d][0]
            j += dir[d][1]
        if 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == color:
            while True:
                i -= dir[d][0]
                j -= dir[d][1]
                if i == x and j == y:
                    break
                valid = True
                board[i][j] = color
    return valid

cdef bint check(int board[8][8], int x, int y, int color):
    cdef int cnt, d, i, j
    for d in range(8):
        i = x
        j = y
        cnt = 0
        while True:
            i += dir[d][0]
            j += dir[d][1]
            if i < 0 or i > 7 or j < 0 or j > 7:
                cnt = 0
                break
            if board[i][j] == -color:
                cnt += 1
            elif board[i][j] == 0:
                cnt = 0
                break
            else:
                break
        if cnt != 0:
            return True
    return False

def get_actions(np.ndarray[DTYPE_t, ndim=2] board, int color):
    cdef int cboard[8][8]
    cdef int i, j, id, ii, jj, k
    for ii in range(8):
        for jj in range(8):
            cboard[ii][jj] = board[ii][jj]
    ret = []
    for id in range(64):
        i = int(id / 8)
        j = id % 8
        if cboard[i][j] == 0:
            if check(cboard, i, j, color):
                ret.append((i,j))
    return ret

# def get_actions(np.ndarray[DTYPE_t, ndim=2] board, int color):
#     cdef int cboard[8][8]
#     cdef int new_cboard[8][8]
#     cdef int i, j, id, ii, jj, k
#     for ii in range(8):
#         for jj in range(8):
#             cboard[ii][jj] = board[ii][jj]
#     ret = []
#     for id in range(64):
#         i = int(id / 8)
#         j = id % 8
#         if cboard[i][j] == 0:
#             for ii in range(8):
#                 for jj in range(8):
#                     new_cboard[ii][jj] = cboard[ii][jj]
#             if cplace(new_cboard, i, j, color):
#                 ret.append((i,j))
#     return ret

# 随机产生决策
def rand_place(np.ndarray[DTYPE_t, ndim=2] board, int color):
    moves = get_actions(board, color)
    if len(moves) == 0:
        return -1, -1
    return random.choice(moves)

cdef cget_actions(int cboard[8][8], int color):
    ret = []
    for id in range(64):
        i = int(id / 8)
        j = id % 8
        if cboard[i][j] == 0:
            if check(cboard, i, j, color):
                ret.append((i,j))
    return ret
def is_terminal(np.ndarray[DTYPE_t, ndim=2] board, int color):
    cdef int cboard[8][8]
    cdef int i, j, id, ii, jj, k
    for ii in range(8):
        for jj in range(8):
            cboard[ii][jj] = board[ii][jj]
    if(len(cget_actions(cboard, color)) ==0 ):
        if(len(cget_actions(cboard, -color)) == 0):
            return True
    return False

