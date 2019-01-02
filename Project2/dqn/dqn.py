import numpy as np

def DQN(S, A, N, M, alpha, gamma):
    """
    训练神经网络Q
    :param S: 状态集
    :param A: 行动集
    :param N: 记忆池容量
    :param M: 训练迭代次数
    :param alpha: 学习率
    :param gamma: 长期回报衰减率
    :return: Q 训练得到的Q
    """
    # 初始化神经网络Q，R矩阵，容量为N的记忆池
    D = init_D(N)
    Q = init_Q(S,A)
    R = init_R(S,A)
    # 迭代M次
    for _ in range(M):
        # 随机选取一个初始状态s
        s = choose_state(S)
        # 得带直到s是目标状态
        while not is_terminal(s):
            # 采用一定策略（如eps-greedy）选择一个行动a
            a = choose_action(A)
            # 采取行动a，得到新状态new_s
            new_s = transfer(s, a)
            # 将s, new_s, R[s,a], a加入记忆池
            D.append([s,a,R[s,a],new_s])
            # 采用minibatch法采样部分样本
            sub_D = minibatch(D)
            # 对于每个样本
            for s, a, r, new_s in D:
                # 为样本设置标签
                y = 0
                # 若是结束态
                if is_terminal(s):
                    y = r
                else:
                    y = r + gamma * np.max(Q[new_s, :])
                # 输入标签值和估计值，依据梯度下降法更新网络
                gradient_descent(y, Q[s, a])
    return Q

def QDecision(Q,A,s):
    """
    使用Q矩阵进行决策
    :param Q: 使用QLearning学习到的Q矩阵
    :param A: 行动集
    :param s: 当前状态
    :return: 当前状态下的最后决策
    """
    # 选择Q(s,a)最大的行动a
    return A[np.argmax(Q[s,:])]

def init_Q(S, A):
    #TODO
    return np.array([[]])

def init_D(S):
    #TODO
    return []

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

def minibatch(D):
    # TODO
    return [[]]

def gradient_descent(y, predict):
    # TODO
    pass
