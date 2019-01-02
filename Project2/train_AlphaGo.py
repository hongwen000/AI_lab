import AlphaGo.Arg as Arg
import mxnet as mx
from mxnet import gluon
from AlphaGo import Network
from play_AlphaGo_vs_AlphaGo import get_mxnet_dataset
import os
myctx = mx.gpu()
net = Network.NN()
net.initialize(ctx=myctx)
fn = "save/{}.param"
for iters in range(30):
    print("iter:", iters)
    os.system("mkdir -p save")
    if iters != 0:
        net.load_parameters(fn.format(iters - 1))
    trainer = gluon.Trainer(net.collect_params(),
                            'sgd', {'wd': Arg.L2_c, 'momentum': 0.9})
    D = get_mxnet_dataset(10, 100, net, myctx)
    Network.train(D, net, Network.loss, trainer, 100, 32, Network.lr, ctx=myctx)
    net.save_parameters(fn.format(iters))
