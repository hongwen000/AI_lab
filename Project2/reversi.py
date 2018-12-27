'''
    黑白棋（Reversi）样例程序
    随机策略
    作者：林舒
    游戏信息：http://www.botzone.org/games#Reversi
'''

import json
import numpy as np
import random

EMPTY = 0
BLACK = 1
WHITE = -1

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

def get_actions(board, color):
    moves = []
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:
                newBoard = board.copy()
                if place(newBoard, i, j, color):
                    moves.append((i, j))
    return moves

# 随机产生决策
def randplace(board, color):
    x = y = -1
    moves = get_actions(board, color)
    if len(moves) == 0:
        return -1, -1
    return random.choice(moves)

# 处理输入，还原棋盘
def initBoard():
    board = np.zeros((8, 8), dtype=np.int)
    board[3][4] = board[4][3] = 1
    board[3][3] = board[4][4] = -1
    myColor = 1
    return board, myColor

def inBoundary(x: int, y: int)->bool:
    return (x >= 0) and (x < 8) and (y >= 0) and (y < 8)
def getXY(i):
    return int(i / 8), i % 8
def getID(x, y):
    return x * 8 + y

def sameColorChess(c1: int, c2: int):
    if(c1 == EMPTY or c2 == EMPTY):
        return False
    if (c1 == c2):
        return True
    return False

def diffColorChess(c1: int, c2: int):
    if(c1 == EMPTY or c2 == EMPTY):
        return False
    if (c1 != c2):
        return True
    return False

grid = np.array([[ 1, 1, 1, 1, 0, 0, 0, 0],
                 [ 1,-1, 1, 1, 1,-1, 0, 0],
                 [ 1,-1, 1, 1, 1,-1,-1,-1],
                 [ 1,-1, 1, 1, 1,-1,-1,-1],
                 [ 1,-1,-1, 1, 1, 1,-1,-1],
                 [ 1,-1,-1, 1, 1, 1,-1,-1],
                 [ 1, 0, 1, 1, 1, 1, 0, 0],
                 [ 1, 1, 1, 1, 1, 1, 0, 0]])



# board, myColor = initBoard()
# x, y = randplace(board, myColor)
# print(json.dumps({"response": {"x": x, "y": y}}))


for i in range(10000):
    get_actions(grid, BLACK)

