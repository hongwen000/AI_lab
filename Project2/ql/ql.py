import numpy as np
def init_Q(S, A):
    #TODO
    return np.array([[]])

def init_R(S, A):
    #TODO
    return np.array([[]])

def is_coverage(Q)->bool:
    #TODO
    return False

def choose_state(S):
    #TODO
    pass

def is_terminal(s)->bool:
    #TODO
    return False

def choose_action(A):
    # TODO
    pass

def transfer(s, a):
    # TODO
    pass

def QLearning(S, A, alpha, gamma):
    """
    训练Q矩阵
    :param S: 状态集
    :param A: 行动集
    :param alpha: 学习率
    :param gamma: 长期回报衰减率
    :return: Q 训练得到的Q
    """
    Q = init_Q(S,A)
    R = init_R(S,A)
    while not is_coverage(Q):
        s = choose_state(S)
        while not is_terminal(s):
            a = choose_action(A)
            new_s = transfer(s, a)
            Q[s, a] = alpha * (R[s,a] + gamma * np.max(Q[new_s,:])) + (1-alpha) * Q[s,a]
            s = new_s

def QDecision(Q,A,s):
    """
    使用Q矩阵进行决策
    :param Q: 使用QLearning学习到的Q矩阵
    :param A: 行动集
    :param s: 当前状态
    :return: 当前状态下的最后决策
    """
    return A[np.argmax(Q[s,:])]
