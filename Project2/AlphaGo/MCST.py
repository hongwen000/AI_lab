from __future__ import annotations
import numpy as np
from numpy.core.multiarray import ndarray
import random

from fast_place import get_actions
from fast_place import place
from typing import *
from math import sqrt
import pygraphviz as gv
import tqdm
import AlphaGo.Arg

arg_C = sqrt(2)

class Node:
    z: Union[None, int]
    board: Union[ndarray, ndarray]
    par: Node
    C: Dict[tuple, Node]
    A: List[tuple]

    def __init__(self, board: np.ndarray, color: int, par: Optional[Node], a_idx: Optional[int]):
        self.board = board
        self.color = color
        # 可能的行动
        self.A = []
        # 子节点访问次数
        self.N = []
        # total action value
        self.W = []
        # mean action value
        self.Q = []
        # prior probability
        self.P = []
        # 存储子节点
        self.C = {}
        # 神经网络估值
        self.v = 0
        # 记录父节点，用于反馈
        self.par = par
        # 记录父节点产生自己的action的下标
        self.pi = a_idx
        # 终点节点
        self.z: Union[None, int] = None

Tree = Node

def is_terminal(node: Node):
    if(len(get_actions(node.board, node.color)) ==0 ):
        if(len(get_actions(node.board, -node.color)) == 0):
            return True
    return False

def networkF(s: Node):
    n = len(s.A)
    return [random.random() for _ in range(n)], 0.8


def expand(cur: Node) -> Node:
    A = get_actions(cur.board, cur.color)
    if len(A) == 0:
        cur.A = [(-1,-1)]
    else:
        cur.A = A
    cur.N = [0 for _ in cur.A]
    cur.W = [0 for _ in cur.A]
    cur.Q = [0 for _ in cur.A]
    cur.P, cur.v = networkF(cur)
    return cur

def get_value(node: Node):
    values = node.Q + AlphaGo.Arg.C * np.array(node.P) * np.sqrt(np.sum(node.N)/(1.0+np.array(node.N)))
    return values

def add_child(node: Node, a, a_idx):
    new_board = node.board.copy()
    place(new_board, a[0], a[1], node.color)
    new_color = -node.color
    new_node = Node(new_board, new_color, node, a_idx)
    node.C[a] = new_node
    return new_node

def best_child(node: Node):
    value = get_value(node)
    a_idx = int(np.argmax(value))
    a = node.A[a_idx]
    if a not in node.C.keys():
        add_child(node, a, a_idx)
    return node.C[a]

def judge(board: np.ndarray, color: int)->int:
    bcnt = np.sum(np.equal(board, 1))
    wcnt = np.sum(np.equal(board, -1))
    if(bcnt == wcnt):
        return 0
    if ((bcnt > wcnt) == (color == 1)):
        return 1
    else:
        return -1


def tree_walk(cur: Node)->Node:
    """
    实现算法的选择步和扩展步
    :param cur: 当前节点
    :return: 返回扩展的新节点
    """
    # 若当前状态不是游戏结束状态
    while not is_terminal(cur):
        # 如果是叶子节点
        if len(cur.A) == 0:
            return expand(cur)
        else:
            cur = best_child(cur)
    if(len(cur.A) == 0):
        expand(cur)
        # add_child(cur, (-1,-1), 0)
        cur.z = judge(cur.board, cur.color)
    return cur

def feedback(n: Node, v: int)->None:
    while n.par is not None:
        n.par.N[n.pi] += 1
        n.par.W[n.pi] += v
        n.par.Q[n.pi] = n.par.W[n.pi] / n.par.N[n.pi]
        n.par.z = n.z
        n = n.par


def best_action(node: Node):
    return node.A[int(np.argmax(np.power(node.N, 1.0 / AlphaGo.Arg.tao)/ (1 + np.power(np.sum(node.N), AlphaGo.Arg.tao))))]

def MCTS(root: Node, N):
    """
    蒙特卡洛树搜索
    :param root: 搜索的根节点
    :param N: 采样次数
    :return: 返回从当前根节点开始的最佳行动
    """
    for _ in tqdm.trange(N):
    # for i in range(N):
        # 算法第1，2步，扩展新节点
        new_node= tree_walk(root)
        # 算法第3，4步，快速搜索到游戏结束，反馈新节点的价值
        feedback(new_node, new_node.v)
    # 最佳子节点
    return best_action(root)

vex = 0
def draw_tree_worker(n: Node, G):
    global vex
    for a, c in n.C.items():
        vex += 1
        num = np.sum(c.N)
        if num == 0:
            G.add_node(id(c), label="Leaf", shape="record")
        else:
            G.add_node(id(c), label="{}".format(np.sum(c.N)), shape="record")
        G.add_edge(id(c.par), id(c), label="{:.1f},{:.1f}".format(n.Q[c.pi], n.P[c.pi]))
        draw_tree_worker(c, G)

def draw_tree(T: Tree, filen = 'alg.svg'):
    G = gv.AGraph()
    G.add_node(id(T), label="Root: {}".format(np.sum(T.N)), shape="record")
    draw_tree_worker(T, G)
    G.node_attr['shape'] = 'circle'
    G.edge_attr['color'] = 'red'
    G.layout(prog='dot')
    G.draw(filen)
    print(vex)
    # img = mpimg.imread('first_pygraphviz.jpg')
    # imgplot = plt.imshow(img)
    # plt.show()
