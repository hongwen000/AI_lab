#include <random>
#include <iostream>
#include "array"
#include "algorithm"
#include "vector"
#include "cstdlib"
#define USE_DEBUG
using namespace std;
#define GENE_CNT 13
using chrom_t = array<double, GENE_CNT>;

std::vector<chrom_t> random_gene(int N)
{
    std::vector<chrom_t > ret;
    srand(time(nullptr));
    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j < GENE_CNT; ++j)
        {
            ret[i][j] = random() % 20 - 10;
        }
    }
    return ret;
}

std::vector<chrom_t> group;
std::vector<chrom_t> next_group;
std::vector<int> score;
int iter = 0;
bool end_evolution(int N, int round)
{
    double total_pk = round * N;
#ifdef USE_DEBUG
    if(score.empty())
        cout << "Begin" << endl;
    else
    {
        cout << "Iter " << ++iter << " result: ";
        for(int i = 0; i < N; ++i)
        {
            cout << score[i] / total_pk << ", ";
        }
        cout << endl;
        cout << "  Best is " << *max_element(score.begin(), score.end()) / total_pk << endl;
    }
#endif
    return iter > 10000;
}

/*
 * 适应度函数
 * 采用循环赛制
 * round: 比赛轮数
 * 返回值：每个染色体的适应度
 */
std::vector<int> calc_fit(int N, int round)
{
    score.clear();
    for(int i = 0; i < N; ++i)
    {
        score.push_back(0);
    }
    for(int k = 0; k < round; ++k)
    {
        for(int i = 0; i < N; ++i)
        {
            for(int j = 0; j < N; ++j)
            {
                auto ret = pk(group[i], group[j]);
                score[i] += ret;
                score[j] -= ret;
            }
        }
    }
    auto m = std::min_element(score.begin(), score.end());
    if(*m < 0)
        for(auto&i: score) i -= *m;
    return score;
}

/*
 * 染色体交叉
 * p1: 双亲之一
 * p2: 双亲之一
 * 返回值： 交叉的到的新染色体
 */
chrom_t cross_chrom(const chrom_t& p1, const chrom_t& p2)
{
    std::vector<int> v;
    for(int i = 0; i < GENE_CNT; ++i)
    {
        v.push_back(i);
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(v.begin(), v.end(), g);
    std::uniform_int_distribution<> dis(0, GENE_CNT);
    auto sp = dis(g);
    chrom_t new_gene;
    for(int i = 0; i < sp; ++i)
    {
        new_gene[v[i]] = p1[v[i]];
    }
    for(int i = sp; i < GENE_CNT; ++i)
    {
        new_gene[v[i]] = p2[v[i]];
    }
    return new_gene;
}

/*
 * 基因突变
 * gene: 进行突变的基因
 * R: 突变概率
 * D: 最大突变程度
 */
void mutate_gene(chrom_t& gene, double R, double D)
{
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0, 1.0);
    std::uniform_real_distribution<> dis2(-1.0, 1.0);
    for(int i = 0; i < GENE_CNT; ++i)
    {
        if(dis(gen) < R)
        {
            gene[i] += dis2(gen) * D * gene[i];
        }
    }
}
/*
遗传算法
:param N: 种群数量
:param M: 克隆数量
:param R: 基因突变率
:param round: 执行循环赛测试适应度的轮数
:return: 生成的最优染色体
 */
chrom_t genetic(int N, int M, double R, double D, int round)
{
    if(N < 2)
    {
        throw(std::exception{});
    }
    group = random_gene(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    while (!end_evolution(N, round))
    {
        auto fitness = calc_fit(N, round);
        next_group.clear();
        std::discrete_distribution<> d(fitness.begin(), fitness.end());
        for(int i = 0; i < M; ++i)
            next_group.push_back(group[d(gen)]);
        for(int i = 0; i < (N-M); ++i)
        {
            auto idx1 = d(gen);
            auto idx2 = d(gen);
            while (idx1 == idx2)
            {
                idx2 = d(gen);
            }
            auto& p1 = group[idx1];
            auto& p2 = group[idx2];
            auto new_gene = cross_chrom(p1, p2);
            mutate_gene(new_gene, R, D);
        }
    }
    return group[distance(max_element(score.begin(), score.end()), score.begin())];
}
