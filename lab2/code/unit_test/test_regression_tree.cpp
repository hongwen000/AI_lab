//
// Created by lixinrui on 10/17/18.
//

#include "AI_utility.h"
#include <fstream>
#include "gtest/gtest.h"
#include "iostream"
#include "regression_tree.h"
#include <chrono>
using namespace std;
TEST(T, 3)
{
Eigen::MatrixXd a(5, 14);
a <<0,0,1,2,2,2,1,0,0,2,0,1,1,2,
2,2,2,1,0,0,0,1,0,1,1,1,2,1,
0,0,0,0,1,1,1,0,1,1,1,0,1,0,
0,1,0,0,0,1,1,0,0,0,1,1,0,1,
0,0,1,1,1,0,1,0,1,1,1,1,1,0;

RegressionTree t(a);
t.train(ErrFunc::var);
std::vector<std::string> mp = {"年龄","收入","学生？","信用等级"};
auto ret = t.predict(a);
for(auto i : ret)
cout << i << ",";
cout << endl;

cout << t.vaild(a) << endl;

cout << t.print(mp) << endl;
}

vector<map<string, int>> mp = {
        {
                {"low", 0},
                {"med", 1},
                {"high", 2},
                {"vhigh", 3}
        },
        {
                {"low", 0},
                {"med", 1},
                {"high", 2},
                {"vhigh", 3}
        },
        {
                {"2", 0},
                {"3", 1},
                {"4", 2},
                {"5more", 3}
        },
        {
                {"2", 0},
                {"4", 1},
                {"more", 2}
        },
        {
                {"small", 0},
                {"med", 1},
                {"big", 2}
        },
        {
                {"low", 0},
                {"med", 1},
                {"high", 2}
        },
        {
                {"0", 0},
                {"1", 1},
                {"��", 0}
        }
};

TEST(T, 4)
{

    auto now = [](){return chrono::steady_clock::now();};
    //auto print_ret = [](auto K, auto name, auto diff, auto acc){cout << "K = " << K << " " << name << " spent " << chrono::duration <double, milli> (diff).count() << " ms, acc: " << acc << endl;};
    auto print_ret = [](auto K, auto name, auto diff, auto acc){cout <<  K << "," << name << "," << chrono::duration <double, milli> (diff).count() << "," << acc << endl;};
    //读取训练集数据
    auto f = readFile("../data/Car_train.csv");
    //向量化
    auto data = vectorizeData(f, mp);
    //K遍历2到8的值
    for(double K: range(2, 9))
    {
        int N = data.cols();
        int pieceSize = N / K;
        auto all = range(0, N);
        double diff = 0;
        double acc = 0;
        for (auto i : range(0, K)) {
            //分割验证集 1/K 的数据作为验证集
            auto vaildSetRange = range(i * pieceSize, (i + 1) * pieceSize);
            matrix_view<double> vaildSet(data, vaildSetRange);
            //其余数据作为训练集
            auto trainSetRange = view::set_difference(all, vaildSetRange);
            matrix_view<double> trainSet(data, trainSetRange);
            RegressionTree t(trainSet);
            auto start = now();
            t.train(ErrFunc::var);
            acc += (t.vaild(vaildSet));
        }
        acc /= K;
        print_ret(K, "regress", diff, acc);
    }
}
int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
