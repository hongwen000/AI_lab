import AlphaGo.Arg as Arg
import time
from AlphaGo.MCTS import *
import matplotlib
from IPython import display
from matplotlib import pyplot as plt
from play_AlphaGo_vs_AlphaGo import init_board
from play_AlphaGo_vs_AlphaGo import get_mxnet_dataset
import gluonbook as gb
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, data as gdata
from AlphaGo.MCTS import to_NN_input

# 残差模块
class Residual(nn.Block):
    def __init__(self, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(channels=Arg.ch, kernel_size=3, strides=1, padding=1)
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(channels=Arg.ch, kernel_size=3, strides=1, padding=1)
        self.bn2 = nn.BatchNorm()

    def forward(self, X:nd.NDArray):
        return nd.relu(X + self.bn2(self.conv2(nd.relu(self.bn1(self.conv1(X))))))

# Alpha zero 神经网络实现
class NN(nn.Block):
    def __init__(self, **kwargs):
        super(NN, self).__init__(**kwargs)
        #
        self.cov1 = nn.Conv2D(channels=Arg.ch, kernel_size=3, strides=1, padding=1)
        self.bn1 = nn.BatchNorm()
        self.resi = nn.Sequential()
        for _ in range(19):
            self.resi.add(Residual(**kwargs))
        self.covp = nn.Conv2D(channels=2, kernel_size=1, strides=1)
        self.bnp = nn.BatchNorm()
        self.fcp = nn.Dense(8*8+1)

        self.covv = nn.Conv2D(channels=1, kernel_size=1, strides=1)
        self.bnv = nn.BatchNorm()
        self.fcv1 = nn.Dense(256)
        self.fcv2 = nn.Dense(1)

    def forward(self, X: nd.NDArray):
        out = self.resi(nd.relu(self.bn1(self.cov1(X))))
        p = self.fcp(nd.relu(self.bnp(self.covp(out))))
        v = self.fcv2(nd.relu(self.fcv1(nd.relu(self.bnv(self.covv(out))))))
        return nd.tanh(v), nd.softmax(p)

#         self.cov2 = nn.Conv2D(channels=Arg.ch, kernel_size=3, activation='relu', strides=1, padding=1)
#         self.bn2 = nn.BatchNorm()
#         self.cov3 = nn.Conv2D(channels=Arg.ch, kernel_size=3, activation='relu', strides=1, padding=1)
#         self.bn3 = nn.BatchNorm()
#         self.cov4 = nn.Conv2D(channels=Arg.ch, kernel_size=3, activation='relu', strides=1, padding=1)
#         self.bn4 = nn.BatchNorm()
#         self.fc1 = nn.Dense(1024, activation='relu')
#         self.bn5 = nn.BatchNorm()
#         self.drop1 = nn.Dropout(0.3)
#         self.fc2 = nn.Dense(512, activation='relu')
#         self.bn6 = nn.BatchNorm()
#         self.drop2 = nn.Dropout(0.3)
#         self.fc3 = nn.Dense(8 * 8 + 1)
#         self.fc4 = nn.Dense(1)
#
#     def forward(self, X: nd.NDArray):
#         nn = self.bn1(self.cov1(X))
#         nn = self.bn2(self.cov2(nn))
#         nn = self.bn3(self.cov3(nn))
#         nn = self.bn4(self.cov4(nn))
#         nn = self.drop1(self.bn5(self.fc1(nn)))
#         nn = self.drop2(self.bn6(self.fc2(nn)))
#         p = self.fc3(nn)
#         v = self.fc4(nn)
#         return nd.tanh(v), nd.softmax(p)

# class NN(nn.Block):
#     def __init__(self, **kwargs):
#         super(NN, self).__init__(**kwargs)
#         self.cov1 = nn.Conv2D(channels=Arg.ch, kernel_size=3, activation='relu', strides=1, padding=1)
#         self.bn1 = nn.BatchNorm()
#         self.cov2 = nn.Conv2D(channels=Arg.ch, kernel_size=3, activation='relu', strides=1, padding=1)
#         self.bn2 = nn.BatchNorm()
#         self.cov3 = nn.Conv2D(channels=Arg.ch, kernel_size=3, activation='relu', strides=1, padding=1)
#         self.bn3 = nn.BatchNorm()
#         self.cov4 = nn.Conv2D(channels=Arg.ch, kernel_size=3, activation='relu', strides=1, padding=1)
#         self.bn4 = nn.BatchNorm()
#         self.fc1 = nn.Dense(1024, activation='relu')
#         self.bn5 = nn.BatchNorm()
#         self.drop1 = nn.Dropout(0.3)
#         self.fc2 = nn.Dense(512, activation='relu')
#         self.bn6 = nn.BatchNorm()
#         self.drop2 = nn.Dropout(0.3)
#         self.fc3 = nn.Dense(8 * 8 + 1)
#         self.fc4 = nn.Dense(1)
#
#     def forward(self, X: nd.NDArray):
#         nn = self.bn1(self.cov1(X))
#         nn = self.bn2(self.cov2(nn))
#         nn = self.bn3(self.cov3(nn))
#         nn = self.bn4(self.cov4(nn))
#         nn = self.drop1(self.bn5(self.fc1(nn)))
#         nn = self.drop2(self.bn6(self.fc2(nn)))
#         p = self.fc3(nn)
#         v = self.fc4(nn)
#         return nd.tanh(v), nd.softmax(p)

# board, color = init_board()
# root = Node(board, color, None, None)
# net = NN()
# net.initialize()
# v, P = net(to_NN_input(root, ctx=mx.cpu()))
# v.asnumpy()
# P.asnumpy()



def lr(step: int)->float:
    """
    根据训练步数调整学习率
    :param step:
    :return:
    """
    if step < 200: return 1e-2
    if step < 600: return 1e-3
    return 1e-4


def loss(z:nd.NDArray, v:nd.NDArray, pi: nd.NDArray, p: nd.NDArray):
    """
    损失函数
    :param z:
    :param v:
    :param pi:
    :param p:
    :return:
    """
    t1 = nd.sum((z-v)**2)
    t3 = -nd.sum(pi*nd.log(p))
    return (t1 + t3)/z.shape[0]

def train(D: gluon.data.ArrayDataset, net:nn.Block, loss, train_fn: gluon.Trainer,
          num_epochs:int, batch_size:int, lr:Callable[[int], float], ctx=mx.cpu()):
    """
    训练函数
    :param D: 数据集
    :param net: 被训练网络
    :param loss: 损失函数
    :param train_fn: 优化器
    :param num_epochs: 训练步数上限
    :param batch_size: 批量数
    :param lr: 学习率函数
    :param ctx: 训练设备
    """
    print('training on', ctx)
    data_iter = gdata.DataLoader(D, batch_size, shuffle=True)
    train_l_sum, n = 0, 0
    ls = []
    for epoch in range(num_epochs):
        start = time.time()
        for batch_i, (X, (z, pi)) in enumerate(data_iter):
            X = X.as_in_context(ctx)
            z = z.as_in_context(ctx).astype('float32')
            pi = pi.as_in_context(ctx)
            with autograd.record():
                v, p = net(X)
                l = loss(z, v, pi, p)
            l.backward()
            train_fn.set_learning_rate(lr(epoch))
            train_fn.step(batch_size)
            n += z.size
            train_l_sum += l.asscalar()
            ls.append(l.asscalar())
        print('epoch {:d}, loss {:.4},'
              'time {:.1} sec'
              .format(epoch + 1, (train_l_sum / n),
                 time.time() - start))
    display.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = (3.5, 2.5)
    display.set_matplotlib_formats('svg')
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('train.png')


#TODO

# def train_gpu(D: Tuple[nd.NDArray, nd.NDArray, nd.NDArray], net:nn.Block, loss, train_fn: gluon.Trainer,
#           num_epochs:int, batch_size:int, lr:Callable[[int], float], ctx=mx.gpu()):
#     print('training on', ctx)
#     Xs = D[0].as_in_context(ctx)
#     zs = D[1].as_in_context(ctx)
#     pis = D[2].as_in_context(ctx)
#     G = [Xs, zs, pis]
#     train_l_sum, n = 0, 0
#     ls = []
#     for epoch in range(num_epochs):
#         start = time.time()
#         for batch_i, data in enumerate(G):
#             X = data[0]
#             z = data[1]
#             pi = data[2]
#             with autograd.record():
#                 v, p = net(X)
#                 l = loss(z, v, pi, p)
#             l.backward()
#             train_fn.set_learning_rate(lr(epoch))
#             train_fn.step(batch_size)
#             n += z.size
#             train_l_sum += l.asscalar()
#             ls.append(l)
#         print('epoch {:d}, loss {:.4},'
#               'time {:.1} sec'
#               .format(epoch + 1, (train_l_sum / n),
#                       time.time() - start))
#     display.set_matplotlib_formats('svg')
#     plt.rcParams['figure.figsize'] = (3.5, 2.5)
#     display.set_matplotlib_formats('svg')
#     plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.savefig('train.png')
#
#
#
#
#
#

