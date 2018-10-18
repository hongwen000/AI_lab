#include "AI_utility.h"
#include <range/v3/core.hpp>
#include <range/v3/view.hpp>
#include <range/v3/algorithm.hpp>
#include <utility>
#include <vector>
#include "matrix_view.h"
#include <tuple>
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
template<typename T>
    using RowVec = Eigen::Matrix<T, 1, Eigen::Dynamic>;
using ErrFunc_t = std::function<double(const RowVec<double>&, double avg)>;
auto range = [](int l, int r){
    return view::ints(l,r);
};
namespace RegressionArgs {
    constexpr double err_tolerance = 0.1;
};
class ErrFunc {
public:
    static double var(const RowVec<double>& D, double avg) {
        double ret = 0.0;
        for(Eigen::Index i = 0; i < D.size(); ++i)
        {
            ret += (avg - D[i]) * (avg - D[i]);
        }
        return ret;
    }
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
        Node* ch_l;
        Node* ch_r;
        //使用的分类特征
        int C = -1;
        //使用的分类值
        double S = INFINITY;
        //是否是叶子节点
        bool isLeaf = false;
        //对应的结果
        double Y;
    };
    Node* root;
    std::tuple<const matrix_view<double>, const matrix_view<double>> split_data(const matrix_view<double>& D, int a, double s);
    void train_worker(Node* node, const ErrFunc_t& errFunc);
    std::string print_worker(Node* node, int n, char cn, const Vec<Str>&, string trace);
    double predict(const Eigen::Matrix<double, Eigen::Dynamic, 1>& X, predictOne);
    int prune_worker(Node* node);
public:
    RegressionTree(const matrix_view<double>& _trainSet);
    void train(const ErrFunc_t& errFunc);
    Vec<double> predict(const matrix_view<double>& X);
    double vaild(const matrix_view<double>& X);
    string print(const Vec<Str>& mp);
    void prune();
};

Eigen::MatrixXd vectorizeData(const FileData_t & fileData, vector<map<string, int>>& mp);
