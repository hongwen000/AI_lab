
#include "regression_tree.h"
#include "cmath"
#include "random_lib.h"

RegressionTree::RegressionTree(const matrix_view<double> &_trainSet)
    : trainSet(_trainSet)
{
    featureCount = (int)trainSet.rows() - 1;
    root = new Node(matrix_view<double>(trainSet), range(0, featureCount));
}


std::tuple<const matrix_view<double>, const matrix_view<double>> RegressionTree::split_data \
    (const matrix_view<double>& D, int a, double s)
{
    auto vals = D.row(a);
    std::vector<Eigen::Index> idx_l;
    std::vector<Eigen::Index> idx_r;
    for(Eigen::Index j = 0; j < vals.size(); ++j)
    {
        if(vals[j] < s)
            idx_l.push_back(j);
        else
            idx_r.push_back(j);
    }

    auto D_l = matrix_view<double>(D, idx_l);
    auto D_r = matrix_view<double>(D, idx_r);
    return std::tie(D_l, D_r);
}

void RegressionTree::train_worker(RegressionTree::Node *node, const ErrFunc_t &errFunc, int depth, int level)
{
    auto* pA = &node->A;
    auto* pD = &node->D;
    auto D_Y = pD->row(-1);
    auto avg = D_Y.sum() / D_Y.size();
    auto err = errFunc(D_Y, avg);
    //cout << "In level " << depth << endl;
    if(err < RegressionArgs::err_tolerance || depth > level)
    {
        node->isLeaf = true;
        node->Y = avg;
        return;
    }
    int bestA = -1;
    double bestS = INFINITY;
    double bestErr = err;
    for(const auto& a : range(0, pA->size()))
    {
        auto vals = pD->row(a);
        std::set<double> vals_set;
        auto n = vals.size();
        //for(Eigen::Index i = 0; i < vals.size(); ++i)
        //{
        //    vals_set.insert(vals[i]);
        //}
        for(Eigen::Index i = 0; i < 10; ++i)
        {
            vals_set.insert(vals[RandLib::uniform_rand(0, n - 1)]);
        }
        for(auto s: vals_set)
        {
            auto[D_l, D_r] = split_data(node->D, a, s);
            auto Y_l = D_l.row(-1);
            auto Y_r = D_r.row(-1);
            auto avg_l = Y_l.sum() / Y_l.size();
            auto avg_r = Y_r.sum() / Y_r.size();
            auto err_l = errFunc(Y_l, avg_l);
            auto err_r = errFunc(Y_r, avg_r);
            auto new_err = err_l + err_r;
            if(new_err < bestErr)
            {
                bestA = a;
                bestS = s;
                bestErr = new_err;
            }
        }
    }
    //cout << "Best cut: new_err = " << bestErr << " , A = " << bestA << " , S = " << bestS << endl;
    if(bestA == -1)
    {
        node->isLeaf = true;
        node->Y = avg;
        return;
    }
    auto[D_l, D_r] = split_data(node->D, bestA, bestS);
    if(D_l.empty())
    {
        node->isLeaf = true;
        node->Y = avg;
        return;
    }
    if(D_r.empty())
    {
        node->isLeaf = true;
        node->Y = avg;
        return;
    }
    node->C = bestA;
    node->S = bestS;
    node->ch_l = new Node(D_l, *pA);
    node->ch_r = new Node(D_r, *pA);
    train_worker(node->ch_l, errFunc, depth+1, level);
    train_worker(node->ch_r, errFunc, depth+1, level);
}

void RegressionTree::train(const ErrFunc_t &errFunc, int level)
{
    train_worker(root, errFunc, 0, level);
}

double RegressionTree::predict(const Eigen::Matrix<double, Eigen::Dynamic, 1>& X, predictOne)
{
    auto ptr = root;
    while(!ptr->isLeaf)
    {
        double v = X(ptr->C, 0);
        if(v < ptr->S)
            ptr = ptr->ch_l;
        else
            ptr = ptr->ch_r;
    }
    return ptr->Y;
}
Vec<double> RegressionTree::predict(const matrix_view<double>& X)
{
    Vec<double> ret;
    if(X.empty()) return ret;
    for(Eigen::Index i = 0; i < X.cols(); ++i)
    {
        ret.push_back(predict(X.col(i), predictOne{}));
    }
    return ret;

}

double RegressionTree::vaild(const matrix_view<double> &vaild_set) {
    auto ret = predict(vaild_set);
    for(auto& i:ret)
    {
        if(i >= 0.5) i = 1;
        else i = 0;
    }
    auto n = vaild_set.cols();
    double correct = 0;
    for(Eigen::Index i = 0; i < n; ++i)
    {
        if((int)vaild_set(-1, i) == (int)ret[i])
            correct++;
    }
    return correct / n;

    auto X = predict(vaild_set);
    auto Y = vaild_set.row(-1);
    auto X_bar = ranges::accumulate(X, 0) * 1.0 / X.size();
    auto Y_bar = Y.sum() / Y.size();
    for(auto& i: X)
        i -= X_bar;
    for(Eigen::Index i = 0; i < Y.size(); ++i)
        Y[i] -= Y_bar;
    double t1, t2, t3;
    t1 = t2 = t3 = 0;
    for(size_t i = 0; i < X.size(); ++i)
        t1 += X[i] * Y[i];
    for(auto i: X)
        t2 += i * i;
    for(Eigen::Index i = 0; i < Y.size(); ++i)
        t3 += Y[i] * Y[i];
    return t1 / std::sqrt(t2 * t3);
}

std::string RegressionTree::print_worker(Node* node, int n, char cn, const Vec<Str>&mp, string trace)
{
    std::stringstream ss;
    string node_name = trace + "F" +  to_string(node->C) + "C" + cn;
    for(auto& i : node_name) if(i == '-') i = '_';
    if(node->isLeaf)
        ss << node_name << "[label=\"" << node->Y << "\"];\n";
    else
        ss << node_name << "[label=\"" << mp[node->C] << "\"];\n";
    if(node->ch_l)
    {
        string child_name = node_name + "F" + to_string(node->ch_l->C) + "CL";
        for(auto& i : child_name) if(i == '-') i = '_';
        ss << node_name << "->" << child_name << "[label=\" < " << node->S << "\"];\n";
        ss << print_worker(node->ch_l, n+1, 'L', mp, node_name);
    }
    if(node->ch_r)
    {
        string child_name = node_name + "F" + to_string(node->ch_r->C) + "CR";
        for(auto& i : child_name) if(i == '-') i = '_';
        ss << node_name << "->" << child_name << "[label=\" >= " << node->S << "\"];\n";
        ss << print_worker(node->ch_r, n+1, 'R', mp, node_name);
    }
    return ss.str();
}

string RegressionTree::print(const Vec<Str>& mp)
{
    std::stringstream s;
    s << "digraph {\n";
    s << print_worker(root, 0, '0', mp, "");
    s << "}\n";
    return s.str();
}

Eigen::MatrixXd vectorizeData(const FileData_t & fileData, vector<map<string, int>>& mp)
{
    if(fileData.empty())
        return Eigen::MatrixXd{};
    Eigen::MatrixXd ret(fileData[0].size(), fileData.size());
    try{
        for(Eigen::Index j = 0; j < ret.cols(); ++j)
        {
            for(Eigen::Index i = 0; i < ret.rows(); ++i)
            {
                if(mp[i].count(fileData[j][i]) == 0)
                    throw(std::runtime_error("Can not translate " + fileData[j][i]));
                ret(i, j) = mp[i][fileData[j][i]];
            }
        }
    }
    catch (const std::runtime_error& e)
    {
        cout << e.what() << endl;
    }
    return ret;
}

matrix_view<double> BaggingRegressTree::sample_D(const matrix_view<double>& trainSet)
{
    std::vector<Eigen::Index> v;
    std::set<Eigen::Index> s;
    auto n = trainSet.cols();
    for(Eigen::Index i = 0; i < n; ++i)
    {
        s.insert(RandLib::uniform_rand(0, n - 1));
    }
    for(auto i : s)
    {
        v.push_back(i);
    }
    return matrix_view<double>(trainSet, v);
}

Vec<Eigen::Index> BaggingRegressTree::sample_A(size_t n, size_t k)
{
    Vec<Eigen::Index> ret;
    if(k >= n)
    {
        return range(Eigen::Index(0), Eigen::Index(n));
    }
    std::set<Eigen::Index> sel;
    while(sel.size() < k)
    {
        int s = RandLib::uniform_rand(0, n - 1);
        sel.insert(s);
    }

    for(auto i : sel)
    {
        ret.push_back(i);
    }
    std::sort(ret.begin(), ret.end());

    return ret;
}

BaggingRegressTree::BaggingRegressTree(const matrix_view<double>& _trainSet)
    :trainSet(_trainSet)
    {
	    featureCount = (int)trainSet.rows() - 1;
    }

void BaggingRegressTree::train(const ErrFunc_t& errFunc, int M, int k, int maxLevel)
{
    for(int i = 0; i < M; ++i)
    {
        matrix_view<double> sub_D = sample_D(trainSet);
        Vec<Eigen::Index> sub_A_idx = sample_A(featureCount, k);
        sub_A_idx.push_back(-1);
        forests_A.push_back(sub_A_idx);
        sub_D.select_row(sub_A_idx);
        RegressionTree* tree = new RegressionTree(sub_D);
        forests.push_back(tree);
        tree->train(errFunc, maxLevel);
    }
}

Vec<double> BaggingRegressTree::predict(const matrix_view<double>& X)
{
    Vec<double> rets(X.cols(), 0.0);
    Vec<double> ret(X.cols(), 0);
    if(X.empty()) return ret;
    for(size_t i = 0; i < forests.size(); ++i)
    {
        auto X_ = X;
        X_.select_row(forests_A[i]);
        auto oneret = forests[i]->predict(X_);
        for(size_t i = 0; i < oneret.size(); ++i)
        {
            rets[i] += oneret[i];
        }
    }
    for(size_t i = 0; i < rets.size(); ++i)
    {
        double n = forests.size();
        ret[i] = rets[i] / n;
        //if(rets[i] > (forests.size() / 2.0))
            //ret[i] = 1;
    }
    return ret;
}

double BaggingRegressTree::vaild(const matrix_view<double>& X)
{
    auto ret = predict(X);
    auto n = X.cols();
    for(auto& i:ret)
    {
        if(i >= 0.5) i = 1;
        else i = 0;
    }
    double correct = 0;
    for(Eigen::Index i = 0; i < n; ++i)
    {
        if(X(-1, i) == ret[i])
            correct++;
    }
    return correct / n;
}
