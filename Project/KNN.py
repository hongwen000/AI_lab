import ai_base
import numpy as np
from typing import List, Tuple, Callable
from tqdm import tnrange, tqdm_notebook

# 一范数
Dis1 = lambda v1, v2: np.linalg.norm(v1 - v2, 1)
# 二范数
Dis2 = lambda v1, v2: np.linalg.norm(v1 - v2, 2)
# 无穷范数
DisInf = lambda v1, v2: np.linalg.norm(v1 - v2, np.inf)
# 余弦距离（1-余弦相关度）
def DisCosine(v1, v2):
    t1 = np.dot(v1,v2)
    t2 = np.linalg.norm(v1)
    t3 = np.linalg.norm(v2)
    ret = 1 - t1 / (t2*t3)
    return ret
def DisInvNormAvg(distances: np.array, Y: np.array) -> np.array:
    '''
    按照归一化的距离倒数加权求和，返回均值
    '''
    # 如果训练集中有向量距离和待预测向量完全一致（距离为0）
    for idx, dis in enumerate(distances):
        if np.isclose(dis, 0):
            # 则直接返回该训练集向量对应的Y
            return Y[idx]
    # 求距离的倒数
    distances = np.array(1.0) / distances
    # 归一化
    s = np.sum(distances)
    distances = distances / s
    # 分别作为权值乘以K个最邻近的训练集向量对应的Y
    tmp = np.diag(distances) @ Y  
    # 加权后Y的个分量求和
    if len(tmp.shape) is 1:
        return tmp
    else:
        return np.sum(tmp,  axis = (0))

def classifyParseY(ydata: List[str], n: int)->np.array:
    '''
    Convert Y data from raw string list to matrix consisted of Y vectors
    e.g.
    ["anger", "disgust", ..., "surprise"] -> 
    |1, 0, 0, 0, 0, 0|
    |0, 1, 0, 0, 0, 0|
    |0, 0, ...,  0, 0|
    |0, 0, 0, 0, 1, 0|
    |0, 0, 0, 0, 0, 1|
    '''
    D = len(ydata)
    
    #fast hash ydata from strings ["anger", "disgust", ...] to [1, 2, ...]^T
    #ydata = np.array(ydata).reshape((-1,1))
    
    '''
    ymat is the column-wise repeat of ydata.
    e.g.
    |0|      |0, 0, 0, 0, 0, 0|
    |1|   -> |1, 1, 1, 1, 1, 1|
    ...      |................|
    |5|      |5, 5, 5, 5, 5, 5|
    ydata -> ymat
    '''
    ymat  = np.tile(ydata, (1, n))
    
    '''
    ycmp is a matrix of which each row is [0, 1, 2, 3, 4, 5]
    |0, 1, 2, 3, 4, 5|
    |0, 1, 2, 3, 4, 5|
    |................|
    |0, 1, 2, 3, 4, 5|
    '''
    ycmp  = np.tile(np.array(range(n)), (D, 1))
    return np.int_(np.equal(ymat, ycmp))

def KNN(trainSet: Tuple[np.array, np.array],
        testVec: np.array,
        DisFunc: Callable[[np.array, np.array], float],
        K: int,
        WeightFunc: Callable[[np.array, np.array], float]) -> np.array: 
    '''
    一个通用的KNN接口
    trainSet: 二元元组，第一个元素是训练集的X，第二个是Y
    testVec: 待预测向量
    DisFunc: 距离函数
    K: K值
    WeightFunc: 依据第一个参数list<距离>,对第二个参数list<Y值>进行加权，返回预测值
    '''
    #对于多个要预测的值，逐一预测
    if len(testVec.shape) > 1:
        n = len(testVec)
        ret = list(range(n))
        for i in tnrange(n):
            ret[i] = KNN(trainSet, testVec[i], DisFunc, K, WeightFunc)
        return np.array(ret)
    else:
        #测量待预测向量到训练集中每个向量的距离
        #distances是一个list<tuple(index, distance)>
        distances = list(enumerate(map(lambda trainVec: DisFunc(trainVec, testVec), trainSet[0])))
        #依据距离从小到大排序
        distances.sort(key=lambda t: t[1])
        #获取最临近的K个训练样本的下标和对应的距离，输出值
        tmp = list(zip(*distances[:K]))
        kNearIdx = list(tmp[0])
        kNearDis = list(tmp[1])
        kNearY   = trainSet[1][kNearIdx, :]
        #对输出值根据距离加权作为预测输出
        return WeightFunc(kNearDis, kNearY)

def get_regress(predictY, vaildY):
    r = [pearsonr(predictY[:, i], vaildY[:, i])[0] for i in range(6)]
    average = np.average(r)
    print("Correlation Coefficient: ", average)
    return average

def get_classify(predictY, vaildY):
    classifyY = np.zeros_like(predictY)
    for i, row in enumerate(predictY):
        m = 0
        idx = 0
        for j, v in enumerate(row):
            if v > m:
                m = v
                idx = j
        classifyY[i][idx] = 1
    ret = np.sum(np.logical_and(classifyY, vaildY)) / vaildX.shape[0]
    print("Classification Accuracy: ", ret)
    return ret

def autoTrain(trainSet: Tuple, vaildSet:Tuple):
    trainX, trainY = trainSet
    vaildX, vaildY = vaildSet
    print("Start training...")
    t = time()
    K_val = range(8, 14)
    DisFuncs = {"Dis1": Dis1, "Dis2": Dis2, "DisInf": DisInf, "DisCosine": DisCosine}
    results_reg = OrderedDict()
    results_cla = OrderedDict()
    for K in K_val:
        for dfname, DisFunc in DisFuncs.items():
            predictY = KNN((trainX,trainY), vaildX, DisFunc, K, DisInvNormAvg)
            cla_ret = get_classify(predictY, vaildY)
            reg_ret = get_regress(predictY, vaildY)
            results_reg[(pfname, K, dfname)] = cla_ret
            results_cla[(pfname, K, dfname)] = reg_ret
            print(pfname, K, dfname, ":", cla_ret, reg_ret)
    print("{} groups of argument tested, spent {}s".format(len(K_val) * len(DisFuncs), time() - t))
    return results

def vaild(trainSet: Tuple, vaildSet: Tuple, K, DisFunc):
    trainX, trainY = trainSet
    vaildX, vaildY = vaildSet
    predictY = KNN(trainSet,vaildX,DisFunc,K,DisInvNormAvg)
    cla_ret = get_classify(predictY, vaildY)
    reg_ret = get_regress(predictY, vaildY)
    print(pfname, K, dfname, ":", cla_ret, reg_ret)
