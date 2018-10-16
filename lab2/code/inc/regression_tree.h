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
class RegressionTree {
    struct predictOne{};
    matrix_view<double> trainSet;
    int featureCount;
    struct Node {
        Node(const matrix_view<double>& _D, const Vec<Index>& _A): D(_D), A(_A){}
        //当前节点的数据集
        matrix_view<double> D;
        //当前节点的特征集
        Vec<Index> A;
        //子节点
        Vec<Node*> child;
        //使用的分类特征
        int C = -1;
        //使用的分类值
        double s;
        //是否是叶子节点
        bool isLeaf = false;
        //对应的结果
        double Y;
    };
    Node* root;
    void train_worker(Node* node, const JudgeFunc_t& judgeFunc);
    std::string print_worker(Node* node, int n, int cn, std::map<int, std::vector<std::string>>& mp, string trace);
    int predict(const Eigen::Matrix<double, Eigen::Dynamic, 1>& X, predictOne);
    int prune_worker(Node* node);
public:
    RegressionTree(const matrix_view<double>& _trainSet);
    void train(const JudgeFunc_t& judgeFunc);
    Vec<int> predict(const matrix_view<double>& X);
    double vaild(const matrix_view<double>& X);
    string print(std::map<int, std::vector<std::string>>& mp);
    void prune();
};

class RegressionJudgeFunc {
    static double H(double p);
    static double gini(double p);
    using EntropyFunc_t = std::function<double(double)>;
    static double JudgeBaseFunc(const matrix_view<int>& D, int feature, int featureVal, const EntropyFunc_t& entropyFunc, bool splitInfoFlag);
public:
    static double ID3(const matrix_view<int>& D, int feature, int featureVal);
    static double C45(const matrix_view<int>& D, int feature, int featureVal);
    static double CART(const matrix_view<int>& D, int feature, int featureVal);
};