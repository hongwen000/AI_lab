'''
    黑白棋（Reversi）样例程序
    随机策略
    作者：林舒
    游戏信息：http://www.botzone.org/games#Reversi
'''

import json
import numpy
import numpy as np
import random
import fast_place

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
                if fast_place.place(newBoard, i, j, color):
                    moves.append((i, j))
    return moves

# 随机产生决策
def rand_place(board, color):
    x = y = -1
    moves = fast_place.get_actions(board, color)
    if len(moves) == 0:
        return -1, -1
    return random.choice(moves)

# 处理输入，还原棋盘
def init_board():
    board = numpy.zeros((8, 8), dtype=numpy.int)
    board[3][4] = board[4][3] = 1
    board[3][3] = board[4][4] = -1
    myColor = 1
    return board, myColor

def print_chess(n):
    if n == 1 :
        return '•'
    elif n == -1:
        return '○'
    else:
        return ' '

def print_game(board, x, y, color, start = False):
    return
    black_score = np.sum(board == 1)
    white_score = np.sum(board == -1)
    if black_score > white_score:
        result = 1
    elif white_score > black_score:
        result = -1
    else:
        result = 0
    if (color == 1):
        str_color = 'black'
    elif (color == -1):
        str_color = 'white'
    else:
        raise NotImplementedError
    if start:
        print("Game start")
    else:
        if (x < 0):
            print("[{}] can't not play !".format(str_color))
        else:
            print("[{}] put the chess piece in [( {} , {})] !!".format(str_color, x, y))
    moves = fast_place.get_actions(board, -color)
    print('------------------------------------')
    print('|   |', end='')
    for j in range(8):
        print(' {} |'.format(str(int(j))), end='')
    print('\n------------------------------------')
    for i in range(8):
        print('| {} |'.format(str(int(i))), end='')
        for j in range(8):
            if (i, j) in moves:
                print(' {} |'.format('+'), end='')
            else:
                print(' {} |'.format(print_chess(board[i][j])), end='')

        print('\n------------------------------------')

    print("[black : {}]".format(black_score))
    print("[white : {}]".format(white_score))
    print('\n')
    return result, black_score + white_score

def print_winner(board):
    black_score = np.sum(board == 1)
    white_score = np.sum(board == -1)
    if black_score > white_score:
        print("black wins")
    elif black_score < white_score:
        print("white wins")
    else:
        print("draw")


def pk():
    board, color = init_board()
    print_game(board, -1, -1, -1, start=True)
    while True:
        x, y = rand_place(board, color)
        if not x == -1:
            place(board, x, y, color)
            print_game(board, x, y, color)
            color = -color
            continue
        else:
            x, y = rand_place(board, color)
            if not x == -1:
                place(board, x, y, color)
                print_game(board, x, y, color)
                color = -color
                continue
            else:
                break
for i in range(10000):
    pk()
