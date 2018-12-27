import numpy as np
class Node:
    def __init__(self, state):
        self.state = state
        self.children = []
    def add_child(self, node):
        self.children.append(node)

Tree = Node

def is_terminal(node: Node):
    # TODO
    return False

def get_avail_action(node: Node):
    # TODO
    return []

def best_child(node: Node):
    # TODO
    return node.children[0]

def transfer_to(node: Node, action):
    # TODO
    return node


def tree_walk(current: Node):
    """
    实现算法的选择步和扩展步
    :param current: 当前节点
    :return: 返回扩展的新节点
    """
    # 若当前状态不是游戏结束状态
    while not is_terminal(current):
        # 获取所有可能行动
        A = get_avail_action(current)
        # 当前子节点数
        num_child = len(current.children)
        if num_child == len(A):
            # 当前节点所有子节点均已探索过
            current = best_child(current)
        else:
            # 还有尚未扩展的子节点，则扩展出该节点
            new_node = transfer_to(current, A[num_child])
            current.children.append(new_node)
            return new_node
    # 处理调用时就是结束状态的情况
    return current

def rush_strategy(current: Node):
    # TODO
    return current

def get_reward(current: Node):
    # TODO
    return 0

def tree_rush(current: Node):
    """
    实现算法的模拟步
    :param current: 当前节点
    :return: 返回新增节点的效用值
    """
    # 若当前状态不是游戏结束状态
    while not is_terminal(current):
        # 采用某种快速决策的方法进行游戏直到结束状态
        current = rush_strategy(current)
    # 返回游戏结束状态估值
    return get_reward(current)

def feedback(new_node: Node, reward):
    # TODO
    pass

def best_action(node: Node):
    # TODO
    pass

def MCTS(root: Node, N):
    """
    蒙特卡洛树搜索
    :param root: 搜索的根节点
    :param N: 采样次数
    :return: 返回从当前根节点开始的最佳行动
    """
    for i in range(N):
        # 算法第1，2步，扩展新节点
        new_node = tree_walk(root)
        # 算法第3，4步，快速搜索到游戏结束，反馈新节点的价值
        reward = tree_rush(new_node)
        feedback(new_node, reward)
    # 最佳子节点
    return best_action(root)
