from typing import *
def random_gene(N: int):
    # TODO
    return [[]]

def end_evolution(group)->bool:
    # TODO
    return False

def calc_fit(gene)->float:
    # TODO
    return 0

def select_gene(group, fitness: List[float]):
    # TODO
    return []

def select_parents(group, fitness: List[float]):
    # TODO
    return [],[]

def cross_gene(parents):
    # TODO
    return []

def mutate_gene(gene, R: float):
    # TODO
    return []

def select_best(group):
    # TODO
    return []

def genetic(N: int, M: int, R: float):
    """
    遗传算法
    :param N: 种群数量
    :param M: 克隆数量
    :param R: 基因突变率
    :return: 生成的最优染色体
    """
    # 随机生成规模为N的种群
    group = random_gene(N)
    while not end_evolution(group):
        # 计算种群中每个个体的适应度
        fitness = [calc_fit(gene) for gene in group]
        next_generation = []
        # 依据适应度挑选M个个体直接复制进入下一代
        for i in range(M):
            next_generation.append(select_gene(group, fitness))
        # 繁殖产生另外N-M个后代
        for i in range(N - M):
            # 挑选双亲
            parents = select_parents(group, fitness)
            # 基因交叉
            new_gene = cross_gene(parents)
            # 基因突变
            new_gene = mutate_gene(new_gene, R)
            next_generation.append(new_gene)
        group = next_generation
    # 返回最佳的基因
    return select_best(group)
