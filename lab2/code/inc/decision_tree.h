//
// Created by 李新锐 on 01/10/2018.
//

#ifndef DECISIONTREE_DECISION_TREE_H
#define DECISIONTREE_DECISION_TREE_H

#include "AI_utility.h"
#include <range/v3/core.hpp>
#include <range/v3/view.hpp>
#include <range/v3/algorithm.hpp>
#include <utility>
#include <vector>
#include "matrix_view.h"
#include "iostream"
using std::vector;
using std::pair;
using std::vector;
using Eigen::Index;
using ranges::v3::max_element;
using namespace ranges;
using std::to_string;
using std::string;
using std::map;
using JudgeFunc_t = std::function<double(const matrix_view<int>&, int feature, int featureVal)>;
auto range = [](int l, int r){
    return view::ints(l,r);
};

class DecisionTree {
    struct predictOne{};
    matrix_view<int> trainSet;
//    const Matrix& trainSet;
    int featureCount;
    const Vec<int>featureValues;
    struct Node {
        Node(const matrix_view<int>& _D, const Vec<Index>& _A): D(_D), A(_A){}
        //当前节点的数据集
        matrix_view<int> D;
        //当前节点的特征集
        Vec<Index> A;
        //子节点
        Vec<Node*> child;
        //使用的分类特征
        int C = -1;
        //是否是叶子节点
        bool isLeaf = false;
        //对应的结果
        int Y;
    };
    Node* root;
    void train_worker(Node* node, const JudgeFunc_t& judgeFunc);
    std::string print_worker(Node* node, int n, int cn, std::map<int, std::vector<std::string>>& mp, string trace);
    int predict(const Eigen::Matrix<int, Eigen::Dynamic, 1>& X, predictOne);
    int prune_worker(Node* node);
public:
    DecisionTree(const matrix_view<int>& _trainSet, const Vec<int>& _featureValues);
    void train(const JudgeFunc_t& judgeFunc);
    Vec<int> predict(const matrix_view<int>& X);
    double vaild(const matrix_view<int>& X);
    string print(std::map<int, std::vector<std::string>>& mp);
    void prune();
};

class JudgeFunc {
    static double H(double p);
    static double gini(double p);
    using EntropyFunc_t = std::function<double(double)>;
    static double JudgeBaseFunc(const matrix_view<int>& D, int feature, int featureVal, const EntropyFunc_t& entropyFunc, bool splitInfoFlag);
public:
    static double ID3(const matrix_view<int>& D, int feature, int featureVal);
    static double C45(const matrix_view<int>& D, int feature, int featureVal);
    static double CART(const matrix_view<int>& D, int feature, int featureVal);
};

Matrix vectorizeData(const FileData_t & fileData, vector<map<string, int>>& mp);

class BaggingTree {
    Vec<DecisionTree*> forests;
    Vec<Vec<Eigen::Index>> forests_A;
    matrix_view<int> trainSet;
    Vec<int> featureValues;
    matrix_view<int> sample_D(const matrix_view<int>& trainSet);
    Vec<Eigen::Index> sample_A(size_t n, size_t k);
public:
    BaggingTree(const matrix_view<int>& _trainSet, const Vec<int>& _featureValues);
    void train(const JudgeFunc_t& judgeFunc, int M, int k);
    Vec<int> predict(const matrix_view<int>& X);
    void prune();
    double vaild(const matrix_view<int>& X);
};


#endif //DECISIONTREE_DECISION_TREE_H
