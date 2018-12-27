'''
    黑白棋（Reversi）样例程序
    随机策略
    作者：林舒
    游戏信息：http://www.botzone.org/games#Reversi
'''

from cython.parallel import prange
import json
import numpy
import numpy as np
import random

DIR = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)) # 方向向量

# 放置棋子，计算新局面
def place(board, x, y, color):
    if x < 0:
        return False
    board[x][y] = color
    valid = False
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

from cpython cimport array
import array

def get_avail(board, color):
    cdef array.array a = array.array('i', [1, 2, 3])
    cdef int[:] ca = a
    for id in prange(64, nogil=True):
        i = id / 8
        j = id % 8
        if board[i][j] == 0:
            newBoard = board.copy()
            if place(newBoard, i, j, color):
                with gil:
                    board[i][j] = id
    return a

