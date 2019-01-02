from __future__ import annotations
import numpy as np
from numpy.core.multiarray import ndarray
import random
from mxnet import nd
from fast_place import get_actions
from fast_place import place
from typing import *
from math import sqrt
import pygraphviz as gv
import tqdm
import AlphaGo.Arg
from scipy.special import softmax

arg_C = sqrt(2)

class Node:
    z: Union[None, int]
    board: Union[ndarray, ndarray]
    par: Node
    C: Dict[tuple, Node]
    A: List[tuple]

    def __init__(self, board: np.ndarray, color: int, par: Optional[Node], a_idx: Optional[int]):
        self.board: np.ndarray = board
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
        # 综合后的概率
        self.pi = []
        # 存储子节点
        self.C = {}
        # 神经网络估值
        self.v = 0
        # 记录父节点，用于反馈
        self.par = par
        # 记录父节点产生自己的action的下标
        self.pidx = a_idx
        # 终点节点
        self.z: Union[None, int] = None

def is_terminal(node: Node):
    if(len(get_actions(node.board, node.color)) ==0 ):
        if(len(get_actions(node.board, -node.color)) == 0):
            return True
    return False

# TODO
# 神经网络实现


def get_value(node: Node):
    """
    获取行动估值
    :param node:
    :return:
    """
    values = node.Q + AlphaGo.Arg.C * np.array(node.P) * np.sqrt(np.sum(node.N)/(1.0+np.array(node.N)))
    # Warning 这儿要加softmax
    node.pi = softmax(values)
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


def feedback(n: Node, winner: int)->None:
    """
    反馈步
    :param n:
    :param winner:
    """
    while n.par is not None:
        n.par.N[n.pidx] += 1
        n.par.W[n.pidx] += n.v
        n.par.Q[n.pidx] = n.par.W[n.pidx] / n.par.N[n.pidx]
        if winner != 0:
            if n.par.color == winner:
                n.par.z = 1
            else:
                n.par.z = -1
        n = n.par



def best_action(node: Node):
    """
    返回最佳行动
    :param node:
    :return:
    """
    return node.A[int(np.argmax(np.power(node.N, 1.0 / AlphaGo.Arg.tao)/ (1 + np.power(np.sum(node.N), AlphaGo.Arg.tao))))]

class MCTS():
    def __init__(self, networkF, ctx):
        self.networkF = networkF
        self.ctx = ctx
        pass

    def expand(self, cur: Node) -> Node:
        A = get_actions(cur.board, cur.color)
        if len(A) == 0:
            cur.A = [(-1,-1)]
        else:
            cur.A = A
        cur.N = [0 for _ in cur.A]
        cur.W = [0 for _ in cur.A]
        cur.Q = [0 for _ in cur.A]
        v, P = self.networkF(to_NN_input(cur, self.ctx))
        npv, npP = v.asnumpy(), P.asnumpy()
        cur.v = float(npv[0])
        for x, y in cur.A:
            if x == -1:
                cur.P.append(float(npP[0][64]))
            else:
                cur.P.append(float(npP[0][x*8 + y]))
        return cur

    def tree_walk(self, cur: Node)->Tuple[Node,int]:
        """
        实现算法的选择步和扩展步
        :param cur: 当前节点
        :return: 返回扩展的新节点
        """
        # 若当前状态不是游戏结束状态
        while not is_terminal(cur):
            # 如果是叶子节点
            if len(cur.A) == 0:
                return self.expand(cur), 0
            else:
                cur = best_child(cur)
        winner = judge(cur.board, cur.color)
        if(len(cur.A) == 0):
            self.expand(cur)
            if winner == cur.color:
                cur.z = 1
            elif winner == -cur.color:
                cur.z = -1
            else:
                cur.z = 0
        return cur, winner

    def Run(self, root: Node, N):
        """
        蒙特卡洛树搜索
        :param root: 搜索的根节点
        :param N: 采样次数
        :return: 返回从当前根节点开始的最佳行动
        """
        for _ in range(N):
        # for i in range(N):
            # 算法第1，2步，扩展新节点
            new_node, winner= self.tree_walk(root)
            # 算法第3，4步，快速搜索到游戏结束，反馈新节点的价值
            feedback(new_node, winner)
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
        G.add_edge(id(c.par), id(c), label="{:.1f},{:.1f}".format(n.Q[c.pidx], n.P[c.pidx]))
        draw_tree_worker(c, G)

def draw_tree(T, filen = 'alg.svg'):
    G = gv.AGraph()
    G.add_node(id(T), label="Root: {}".format(np.sum(T.N)), shape="record")
    draw_tree_worker(T, G)
    G.node_attr['shape'] = 'circle'
    G.edge_attr['color'] = 'red'
    G.layout(prog='dot')
    G.draw(filen)
    # img = mpimg.imread('first_pygraphviz.jpg')
    # imgplot = plt.imshow(img)
    # plt.show()

def to_NN_input(n: Node, ctx)->nd.NDArray:
    s = np.zeros((2,8,8))
    s[0] = n.board
    if n.color == 1:
        s[1] = np.zeros((8,8))
    else:
        s[1] = np.ones((8,8))
    s = nd.array(s).reshape((1,2,8,8)).as_in_context(ctx)
    return s

# board, z, pi

def to_example(n: Node)->Tuple[np.ndarray, int, np.ndarray]:
    s = np.zeros((2,8,8))
    s[0] = n.board
    if n.color == 1:
        s[1] = np.zeros((8,8))
    else:
        s[1] = np.ones((8,8))
    pi = np.zeros((65))
    for i, (x, y) in enumerate(n.A):
        if x == -1:
            pi[64] = 1
        else:
            pi[x * 8 + y] = n.pi[i]
    return s, n.z, pi
