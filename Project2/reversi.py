'''
    黑白棋（Reversi）样例程序
    随机策略
    作者：林舒
    游戏信息：http://www.botzone.org/games#Reversi
'''

import json
import numpy
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

# 随机产生决策
def randplace(board, color):
    x = y = -1
    moves = []
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:
                newBoard = board.copy()
                if place(newBoard, i, j, color):
                    moves.append((i, j))
    if len(moves) == 0:
        return -1, -1
    return random.choice(moves)

# 处理输入，还原棋盘
def initBoard():
    fullInput = json.loads(input())
    requests = fullInput["requests"]
    responses = fullInput["responses"]
    board = numpy.zeros((8, 8), dtype=numpy.int)
    board[3][4] = board[4][3] = 1
    board[3][3] = board[4][4] = -1
    myColor = 1
    if requests[0]["x"] >= 0:
        myColor = -1
        place(board, requests[0]["x"], requests[0]["y"], -myColor)
    turn = len(responses)
    for i in range(turn):
        place(board, responses[i]["x"], responses[i]["y"], myColor)
        place(board, requests[i + 1]["x"], requests[i + 1]["y"], -myColor)
    return board, myColor

from copy import deepcopy
checkDirection= [
    [-1, 0],
    [-1, 1],
    [ 0, 1],
    [ 1, 1],
    [ 1, 0],
    [1, -1],
    [0, -1],
    [-1,-1],
]

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

__id2 = 0
__flag = False
__this_d = 0
def getAvail(state: bytearray, color: int):
    global checkDirection
    ret = deepcopy(state)
    for i in range(len(ret)):
        ret[i] = 0
    for i in range(len(ret)):
        if state[i] != EMPTY:
            ret[i] = 0
            continue
        x, y = getXY(i)
        for d in range(8):
            global __this_d
            __this_d = 0
            dx, dy = checkDirection[d]
            x2 = x + dx
            y2 = y + dy
            global __id2
            global __flag
            __flag = False
            __id2 = getID(x2, y2)
            if(inBoundary(x2, y2) and diffColorChess(color, state[__id2])):
                __this_d += 1
                x2 += dx
                y2 += dy
                while(inBoundary(x2, y2)):
                    __id2 = getID(x2, y2)
                    if sameColorChess(color, state[__id2]):
                        __flag = True
                        break
                    elif diffColorChess(color, state[__id2]):
                        __this_d += 1
                    elif state[__id2] == EMPTY:
                        break
                    x2 += dx
                    y2 += dy

                if not __flag:
                    __this_d = 0
            ret[i] += __this_d
    return ret

def AI(state: bytearray, chesscolor: int):
    r = random.Random()
    r.seed()
    avi = getAvail(state, chesscolor)
    possibleMove = []
    for i in range(64):
        if avi[i] > 0:
            possibleMove.append(i)
    if len(possibleMove) == 0:
        return -1
    n = int((r.random() * 64) % len(possibleMove))
    return possibleMove[n]

board, myColor = initBoard()
x, y = randplace(board, myColor)
print(json.dumps({"response": {"x": x, "y": y}}))


