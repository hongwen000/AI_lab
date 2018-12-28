from __future__ import annotations
import numpy as np
from fast_place import get_actions
from fast_place import place
from fast_place import rand_place
from typing import *
from copy import deepcopy
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pygraphviz as gv
import tqdm
from config import *

arg_C = sqrt(2)
def arg_Explore(node: Node):
    n =  len(node.C)
    if(n == 0):
        return 1
    return 0.6 + arg_C * np.sqrt(n*np.log(n))

class Node:
    def __init__(self, board: np.ndarray, color: int, par: Optional(Node)):
        self.board = board
        self.color = color
        # 当前节点访问次数
        self.n = 0
        # 存储子节点
        self.C: List[Node] = []
        # 记录从当前节点开始的获胜次数
        self.w = 0
        # 记录父节点，用于反馈
        self.par = par
    def add_child(self, node):
        self.C.append(node)

Tree = Node

def is_terminal(node: Node):
    if(len(get_actions(node.board, node.color)) ==0 ):
        if(len(get_actions(node.board, -node.color)) == 0):
            return True
    return False

values_grid = np.array([[ 30, -25, 10, 5, 5, 10, -25,  30,],
                   [-25, -25,  1, 1, 1,  1, -25, -25,],
                   [ 10,   1,  5, 2, 2,  5,   1,  10,],
                   [  5,   1,  2, 1, 1,  2,   1,   5,],
                   [  5,   1,  2, 1, 1,  2,   1,   5,],
                   [ 10,   1,  5, 2, 2,  5,   1,  10,],
                   [-25, -25,  1, 1, 1,  1, -25, -25,],
                   [ 30, -25, 10, 5, 5, 10, -25,  30,]])

def get_value(node: Node):
    if len(node.C) == 0:
        return 0
    W = np.array([c.w for c in node.C])
    N = np.array([c.n for c in node.C])
    t = np.log(np.sum(N))
    value = W / N + arg_C * np.sqrt(t/N)
    return value

def get_value_heru(node: Node):
    if len(node.C) == 0:
        return 0
    W = np.array([c.w for c in node.C])
    N = np.array([c.n for c in node.C])
    t = np.log(np.sum(N))
    heru = (values_grid * node.board).sum()
    value = heru + W / N + arg_C * np.sqrt(t/N)
    return value

def best_child(node: Node):
    value = get_value(node)
    b = int(np.argmax(value))
    if b >= len(node.C):
        raise Exception
    return node.C[b]

def transfer_to(node: Node, action):
    new_board = node.board.copy()
    vaild = place(new_board, action[0], action[1], node.color)
    new_color = -node.color
    new_node = Node(new_board, new_color, node)
    return new_node


def tree_walk(cur: Node):
    """
    实现算法的选择步和扩展步
    :param cur: 当前节点
    :return: 返回扩展的新节点
    """
    # 若当前状态不是游戏结束状态
    while not is_terminal(cur):
        # 获取所有可能行动
        A = get_actions(cur.board, cur.color)
        # 当前子节点数
        num_child = len(cur.C)
        # 处理弃局的情况
        if len(A) == 0:
            # 已经生成子节点
            if num_child == 1:
                cur = cur.C[0]
            else:
                # 没有生成子节点
                new_node = transfer_to(cur, (-1,-1))
                cur.C.append(new_node)
                return new_node
        else:
            if num_child == len(A):
                # 当前节点所有子节点均已探索过
                cur = best_child(cur)
            else:
                # 还有尚未扩展的子节点，则扩展出该节点
                new_node = transfer_to(cur, A[num_child])
                cur.C.append(new_node)
                return new_node
    # 处理调用时就是结束状态的情况
    return cur

def tree_walk_deep(cur: Node):
    """
    实现算法的选择步和扩展步
    :param cur: 当前节点
    :return: 返回扩展的新节点
    """
    # 若当前状态不是游戏结束状态
    while not is_terminal(cur):
        # 获取所有可能行动
        A = get_actions(cur.board, cur.color)
        # 当前子节点数
        num_child = len(cur.C)
        # 处理弃局的情况
        if len(A) == 0:
            # 已经生成子节点
            if num_child == 1:
                cur = cur.c[0]
            else:
                # 没有生成子节点
                new_node = transfer_to(cur, (-1,-1))
                cur.C.append(new_node)
                return new_node
        else:
            if num_child == len(A):
                # 当前节点所有子节点均已探索过
                cur = best_child(cur)
            else:
                v = np.max(get_value_heru(cur))
                if v < arg_Explore(cur):
                    # 还有尚未扩展的子节点，则扩展出该节点
                    new_node = transfer_to(cur, A[num_child])
                    cur.C.append(new_node)
                    return new_node
                else:
                    cur = best_child(cur)
    # 处理调用时就是结束状态的情况
    return cur
def rush_strategy(cur: Node):
    x, y = rand_place(cur.board, cur.color)
    place(cur.board, x, y, cur.color)
    cur.color = -cur.color
    return cur

def get_winner(final: Node, color):
    black_score = np.sum(final.board == 1)
    white_score = np.sum(final.board == -1)
    if black_score > white_score:
        return 1
    else:
        return -1


def tree_rush(node: Node):
    """
    实现算法的模拟步
    :param node: 当前节点
    :return: 返回新增节点的效用值
    """
    # 若当前状态不是游戏结束状态
    copy_color: int = node.color
    copy_color2: int = node.color
    cur = Node(node.board.copy(), copy_color, None)
    while not is_terminal(cur):
        # 采用某种快速决策的方法进行游戏直到结束状态
        cur = rush_strategy(cur)
    # 返回游戏结束状态估值
    if(node.color != copy_color2):
        raise Exception
    return get_winner(cur, node.color)

def feedback(node: Node, reward):
    while node is not None:
        node.n += 1
        node.w += reward
        node = node.par


def best_action(node: Node):
    A = get_actions(node.board, node.color)
    if len(A) == 0:
        if define_debug:
            print("pass")
        return (-1,-1)
    else:
        W = [c.w for c in node.C]
        idx = int(np.argmax(W))
        if define_debug:
            print("best choice is the child {}, with wining rate {}".format(idx, W[idx] / node.C[idx].n))
        return A[idx]

def MCTS(root: Node, N):
    """
    蒙特卡洛树搜索
    :param root: 搜索的根节点
    :param N: 采样次数
    :return: 返回从当前根节点开始的最佳行动
    """
    for i in tqdm.trange(N):
    # for i in range(N):
        # 算法第1，2步，扩展新节点
        new_node = tree_walk(root)
        # 算法第3，4步，快速搜索到游戏结束，反馈新节点的价值
        winner = tree_rush(new_node)
        if winner == root.color:
            reward = 1
        else:
            reward = 0
        feedback(new_node, reward)
    # 最佳子节点
    return best_action(root)

def draw_tree_worker(node: Node, G):
    G.add_node(id(node), label= "{}/{}".format(node.w, node.n))
    for c in node.C:
        draw_tree_worker(c, G)
        G.add_edge(id(node), id(c))

def draw_tree(T: Tree):
    G = gv.AGraph()
    draw_tree_worker(T, G)
    G.node_attr['shape'] = 'circle'
    G.edge_attr['color'] = 'red'
    G.layout(prog='dot')
    G.draw('3.svg')
    # img = mpimg.imread('first_pygraphviz.jpg')
    # imgplot = plt.imshow(img)
    # plt.show()
