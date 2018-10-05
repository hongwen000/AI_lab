//
// Created by 李新锐 on 03/10/2018.
//

#include "decision_tree.h"


void DecisionTree::train_worker(DecisionTree::Node *node, std::function<double (const matrix_view<int> &, int, int)> &GainFunc)
{
    auto D_Y = node->D.row(-1);
//    std::cout << "-------------" << std::endl;
//    std::cout << "[D_Y]: " << D_Y << std::endl;
    auto mode = D_Y.sum() >= (node->D.cols() / 2);
    //若A为空集
    if(node->A.empty())
    {
        node->isLeaf = true;
        node->Y = mode;
        return;
    }
    //若D中全部样本属于同一类型
    if(all_equal(D_Y))
    {
        node->isLeaf = true;
        node->Y = D_Y[0];
        return;
    }
    //若所有样本各个属性值均相同
    bool flag = true;
    for(Index i = 0; i < node->D.rows() - 1; ++i)
    {
        if(!all_equal(node->D.row(i)))
        {
            flag = false;
            break;
        }
    }
    if(flag)
    {
        node->isLeaf = true;
        node->Y = mode;
        return;
    }

    Vec<double> info_gain;
    for(auto i: node->A)
        info_gain.push_back(GainFunc(node->D, i, featureValues[i]));
    auto a_star = node->A[max_element(info_gain) - std::begin(info_gain)];
    Vec<Vec<Index>> S(featureValues[a_star]);
    for(int i = 0; i < node->D.cols(); ++i)
    {
        S[node->D(a_star, i)].push_back(i);
    }
    node->C = a_star;
    node->A.erase(ranges::v3::find(node->A, a_star));
    for(int i = 0; i < featureValues[a_star]; ++i)
    {
        auto n = new Node(matrix_view<int>(trainSet, S[i]), node->A);
        if(S[i].empty())
        {
            n->isLeaf = true;
            n->Y = mode;
            node->child.push_back(n);
        }
        else
        {
            train_worker(n, GainFunc);
            node->child.push_back(n);
        }
    }
}

std::string DecisionTree::print_worker(DecisionTree::Node *node, int n, int cn, std::map<int, std::vector<std::string> > &mp, string trace)
{
    std::stringstream ss;
    string node_name = trace + "F" +  to_string(node->C) + "C" + to_string(cn);
    for(auto& i : node_name)
    {
        if(i == '-')
            i = '_';
    }
    if(node->isLeaf)
        ss << node_name << "[label=\"" << mp[node->C][node->Y] << "\"];\n";
    else
        ss << node_name << "[label=\"" << mp[node->C][0] << "\"];\n";

//    for(int i = 0; i < n; ++i)
//        std::cout << '\t';
//    std::cout << node->C << std::endl;
    for(size_t c = 0; c < node->child.size(); ++c)
    {
        if(node->child[c])
        {
            string child_name = node_name + "F" + to_string(node->child[c]->C) + "C" + to_string(c);;
            for(auto& i : child_name)
            {
                if(i == '-')
                    i = '_';
            }
            ss << node_name << "->" << child_name << "[label=\"" << mp[node->C][c + 1] << "\"];\n";
            ss << print_worker(node->child[c], n+1, c, mp, node_name);
        }
    }
    return ss.str();
}

DecisionTree::DecisionTree(const Matrix &_trainSet, const Vec<int> &_featureValues)
    :trainSet(_trainSet),
      featureCount(_featureValues.size()),
      featureValues(_featureValues),
      root(new Node(
               matrix_view<int>(trainSet),
               range(0, featureCount)))
{}

void DecisionTree::train(std::function<double (const matrix_view<int> &, int, int)> &GainFunc)
{
    train_worker(root, GainFunc);
}

string DecisionTree::print(std::map<int, std::vector<std::string> > &mp)
{
    std::stringstream s;
    s << "digraph {\n";
    s << print_worker(root, 0, 0, mp, "");
    s << "}\n";
    return s.str();
}

double JudgeFunc::H(double p)
{
    return -p * std::log2(p) - (1 - p) * std::log2(1 - p);
}

double JudgeFunc::ID3(const matrix_view<int> &D, int feature, int featureVal)
{
    auto D_Y = D.row(-1);
    auto p = D_Y.sum() * 1.0 / D_Y.size();
    auto H_D = H(p);
    Vec<pair<int, int>> S(featureVal);
    int n = D.cols();
    for(int i = 0; i < n; ++i)
    {
        S[D(feature, i)].first++;
        S[D(feature, i)].second += D(-1, i);
    }
    double H_D_A = 0;
    for(auto& p: S)
    {
        //        std::cout << p.first << " ~ " << p.second << std::endl;
        if(p.first == 0 || p.first == p.second)
            continue;
        H_D_A += (p.first * 1.0 / n) * H(p.second * 1.0 / p.first);
        //        std::cout << H_D_A << std::endl;
    }
    return H_D - H_D_A;
}

double JudgeFunc::C45(const matrix_view<int> &D, int feature, int featureVal)
{
    auto D_Y = D.row(-1);
    auto p = D_Y.sum() * 1.0 / D_Y.size();
    auto H_D = H(p);
    Vec<pair<int, int>> S(featureVal);
    int n = D.cols();
    for(int i = 0; i < n; ++i)
    {
        S[D(feature, i)].first++;
        S[D(feature, i)].second += D(-1, i);
    }
    double H_D_A = 0;
    double SplitInfo = 0;
    for(auto& p: S)
    {
        if(p.first == 0 || p.first == p.second)
            continue;
        H_D_A += (p.first * 1.0 / n) * H(p.second * 1.0 / p.first);
        SplitInfo += (-(p.first*1.0/n) * std::log2(p.first*1.0/n));
    }
    auto Gain_D =  H_D - H_D_A;
    return Gain_D / SplitInfo;
}

double JudgeFunc::gini(double p)
{
    return 2 * p * (1 - p);
}

double JudgeFunc::CART(const matrix_view<int> &D, int feature, int featureVal)
{
    auto D_Y = D.row(-1);
    auto p = D_Y.sum() * 1.0 / D_Y.size();
    auto gini_D = gini(p);
    Vec<pair<int, int>> S(featureVal);
    int n = D.cols();
    for(int i = 0; i < n; ++i)
    {
        S[D(feature, i)].first++;
        S[D(feature, i)].second += D(-1, i);
    }
    double gini_D_A = 0;
    for(auto p: S)
    {
        if(p.first == 0 || p.first == p.second)
            continue;
        gini_D_A += (p.first * 1.0 / n) * gini(p.second * 1.0 / p.first);
    }
    return gini_D - gini_D_A;
}

Matrix vectorizeData(const FileData_t & fileData, vector<map<string, int>>& mp)
{
    if(fileData.empty())
        return Matrix{};
    Matrix ret(fileData[0].size(), fileData.size());
    for(Eigen::Index j = 0; j < ret.cols(); ++j)
    {
        for(Eigen::Index i = 0; i < ret.rows(); ++i)
        {
            if(mp[i].count(fileData[j][i]) == 0)
                throw(std::runtime_error("Can not translate " + fileData[j][i]));
            ret(i, j) = mp[i][fileData[j][i]];
        }
    }
    return ret;
}
