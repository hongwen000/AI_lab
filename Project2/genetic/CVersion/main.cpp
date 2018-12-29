#include <random>
#include "set"
#include <cmath>
#include <iostream>
#include "array"
#include "algorithm"
#include "vector"
#include "cstdlib"
#include "fstream"
using namespace std;

#include "reversi.h"

std::vector<chrom_t> group;
std::vector<chrom_t> next_group;
std::vector<int> score;
int iter = 0;
time_t rawtime;
struct tm * timeinfo;
static char time_buffer[80];
std::string timestr;

std::random_device rd;
std::mt19937 gen(rd());

void random_gene(int N)
{
    std::uniform_int_distribution<> dis(-10, 10);

    for(int i = 0; i < N; ++i)
    {
        group.push_back(chrom_t{});
        for(int j = 0; j < GENE_CNT; ++j)
        {
            group[i][j] = dis(gen);
        }
    }
}

bool end_evolution(int N, int round)
{
    double total_pk = round * N;
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
    if(!score.empty())
    {
        vector<size_t> idx(score.size());
        iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(),
             [](auto i1, auto i2) {return score[i1] > score[i2];});
        system(("mkdir -p ret" + timestr).data());
        ofstream f("ret" + timestr + "/" + std::to_string(iter) + ".txt", std::ios::app);
        for(int j = 0; j < N; ++j)
        {
            f << score[idx[j]] << ": ";
            for(auto i = 0; i < GENE_CNT; ++i)
            {
                f << group[idx[j]][i] << ",";
            }
            f << endl;
        }
        f.close();
    }
    return iter > 10000;
}

/*
 * ?????
 * ??????
 * round: ????
 * ?????????????
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
        int draw = 0;
        for(int i = 0; i < N; ++i)
        {
            int sum = 0;
#pragma omp parallel for reduction(+:sum) num_threads(26)
            for(int j = i + 1; j < N; ++j)
            {
                auto ret = pk(group[i], group[j]);
                if(ret == 1)
                {
//                    printf("%d vs %d, %d wins\n", i, j, i);
                    sum+=1;
                }
                else if(ret == -1)
                {
//                    printf("%d vs %d, %d wins\n", i, j, j);
                    score[j]++;
                }
            }
            score[i] += sum;
        }
//        auto ss = std::accumulate(score.begin(), score.end(),draw);
//        if(ss == N * (N-1) / 2)
//            throw(std::runtime_error{"?"});
//        else
//        {
//            cout << ss << " " << N * (N-1) / 2 << endl;
//            throw(std::runtime_error{"??"});
//        }
    }
//    auto m = std::min_element(score.begin(), score.end());
//    if(*m < 0)
//        for(auto&i: score) i -= *m;
    return score;
}

/*
 * ?????
 * p1: ????
 * p2: ????
 * ???? ?????????
 */
chrom_t cross_chrom(const chrom_t& p1, const chrom_t& p2)
{
    std::vector<int> v;
    for(int i = 0; i < GENE_CNT; ++i)
    {
        v.push_back(i);
    }
    std::shuffle(v.begin(), v.end(), gen);
    std::uniform_int_distribution<> dis(0, GENE_CNT);
    auto sp = dis(gen);
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

void normalize(chrom_t & gene)
{
    int too_small = 1;
    for(int i = 0; i < GENE_CNT; ++i)
    {
        if(abs(gene[i]) > 1) too_small--;
    }
    if(too_small >= 0)
    {
        for(int i = 0; i < GENE_CNT; ++i)
            gene[i] *= 10;
    }
    bool too_large = false;
    for(int i = 0; i < GENE_CNT; ++i)
    {
        if(abs(gene[i]) > 100000) too_large = true;
    }
    if(too_large)
    {
        for(int i = 0; i < GENE_CNT; ++i)
            gene[i] /= 10.0;
    }
}
void mutate_gene(chrom_t& gene, double R, double D)
{
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

chrom_t genetic(int N, int M, double R, double D, int round)
{
    if(N < 2 || M > N)
    {
        throw(std::exception{});
    }
    random_gene(N);
    while (!end_evolution(N, round))
    {
        auto fitness = calc_fit(N, round);
        next_group.clear();
        std::discrete_distribution<> d(fitness.begin(), fitness.end());
        std::set<int> choices;
        while (choices.size() < M)
        {
            auto choice = d(gen);
            if(choices.count(choice))
                continue;
            choices.insert(choice);
            normalize(group[choice]);
            next_group.push_back(group[choice]);
        }
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
            normalize(new_gene);
            next_group.push_back(new_gene);
        }
        group = next_group;
    }
    return group[distance(max_element(score.begin(), score.end()), score.begin())];
}

int main()
{
    time (&rawtime);
    timeinfo = localtime(&rawtime);
    timestr = string(time_buffer);
    strftime(time_buffer,sizeof(time_buffer),"%Y-%m-%d-%H-%M-%S",timeinfo);
    timestr = string(time_buffer);
    genetic(26, 3, 0.05, 1, 1);
}