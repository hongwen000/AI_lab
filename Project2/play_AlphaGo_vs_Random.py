import numpy
from typing import *
import numpy as np
import fast_place
import AlphaGo.MCTS as MCTS
from config import *
from mxnet.gluon import data as gdata
from mxnet import nd
from tqdm import trange
import multiprocessing as mp


DIR = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)) # 方向向量

ret = 0
iters = 0
# 放置棋子，计算新局面
def place(board, x, y, color):
    if x < 0:
        return False
    board[x][y] = color
    valid = False
    for d in range(8):
        i = x + DIR[d][0]
        j = y + DIR[d][1]
        while 0 <= i < 8 and 0 <= j < 8 and board[i][j] == -color:
            i += DIR[d][0]
            j += DIR[d][1]
        if 0 <= i < 8 and 0 <= j < 8 and board[i][j] == color:
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

def print_board(board, color):
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

def print_game(board, x, y, color, start = False):
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
    print_board(board, color)
    print("[black : {}]".format(black_score))
    print("[white : {}]".format(white_score))
    print('\n')
    return result, black_score + white_score

def calc_winner(board):
    black_score = np.sum(board == 1)
    white_score = np.sum(board == -1)
    if black_score > white_score:
        if(define_debug):
            print("black wins")
        return 1
    elif black_score < white_score:
        if(define_debug):
            print("white wins")
        return -1
    else:
        if(define_debug):
            print("draw")
        return 0


def pk(N: int, net, ctx) -> MCTS.Node:
    board, color = init_board()
    root = MCTS.Node(board.copy(), color, None, None)
    Game = MCTS.MCTS(net, ctx)
    cur = root
    if(define_debug):
        print_game(board, -1, -1, -1, start=True)
    while True:
        if (color == WHITE):
            x, y = Game.Run(cur, N)
            x, y = fast_place.rand_place(board, color)
        else:
            x, y = Game.Run(cur, N)
        if not x == -1:
            if (x,y) not in cur.C:
                for ii in range(len(cur.A)):
                    if cur.A[ii] == (x,y):
                        MCTS.add_child(cur, (x,y), ii)
                        break
            cur = cur.C[(x,y)]
            fast_place.place(board, x, y, color)
            if(define_debug):
                print_game(board, x, y, color)
            color = -color
            continue
        else:
            if not fast_place.is_terminal(board, color):
                if (color == WHITE):
                    x, y = Game.Run(cur, N)
                    x, y = fast_place.rand_place(board, color)
                else:
                    x, y = Game.Run(cur, N)
                if (x,y) not in cur.C:
                    for ii in range(len(cur.A)):
                        if cur.A[ii] == (x,y):
                            MCTS.add_child(cur, (x,y), ii)
                            break
                cur = cur.C[(x,y)]
                fast_place.place(board, x, y, color)
                if(define_debug):
                    print_game(board, x, y, color)
                color = -color
                continue
            else:
                calc_winner(board)
                break
    return root

def generate_data_worker(node: MCTS.Node, E):
    if node.z is not None and len(node.pi) != 0:
        e = MCTS.to_example(node)
        E.append(e)
    for c in node.C.values():
        generate_data_worker(c, E)

# def generate_data(N: int):
#     Ex = []
#     for i in range(N):
#         E = []
#         root = pk(100)
#         generate_data_worker(root, E)
#         Ex.append(E)
#     return Ex
#
#
# def to_mxnet_dataset(E:List[List[Tuple[np.ndarray, int, np.ndarray]]]):
#     for game in E:
#         l_s, l_z, l_pi = zip(*game)
#         features = np.array(l_s)
#         z = np.array(l_z)
#         pi = np.array(l_pi)
#


# ret = generate_data(1)

def to_mxnet_dataset(Ex:List[Tuple[np.ndarray, int, np.ndarray]]):
    l_s, l_z, l_pi = zip(*Ex)
    features = nd.array(np.array(l_s))
    z = nd.array(np.array(l_z))
    pi = nd.array(np.array(l_pi))
    ret = gdata.ArrayDataset(features, list(zip(z,pi)))
    return ret

def get_mxnet_dataset(play_n: int, mcst_n: int, net, ctx):
    Ex = []
    for i in trange(play_n):
        root = pk(mcst_n, net, ctx)
        generate_data_worker(root, Ex)
    return to_mxnet_dataset(Ex)

# def to_dataset(Ex:List[Tuple[np.ndarray, int, np.ndarray]]):
#     l_s, l_z, l_pi = zip(*Ex)
#     features = nd.array(np.array(l_s))
#     z = nd.array(np.array(l_z))
#     pi = nd.array(np.array(l_pi))
#     return features, z,pi

# def get_dataset(play_n: int, mcst_n: int):
#     return to_dataset(generate_data(play_n, mcst_n))
# ret = generate_data(1)
#
# dataset = to_mxnet_dataset(ret)
# pass
# batch_size = 10
# data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

# for X, y in data_iter:
#     print(X, y)
#     break
import mxnet as mx
from mxnet import nd
from mxnet.gluon import data as gdata
from AlphaGo import Network

myctx = mx.gpu(0)
net = Network.NN()
print("Loading!")
net.load_parameters("pre_train.param", ctx=myctx)
# net.initialize(ctx=myctx)
# fn = "/data/lixr/save/{}.param"
print("Start!")
for i in trange(100):
    pk(25, net, myctx)
    print(ret, iters)

