//
// Created by lixinrui on 10/17/18.
//

#include "AI_utility.h"
#include <fstream>
#include "gtest/gtest.h"
#include "iostream"
#include "regression_tree.h"
#include <chrono>
#include <omp.h>
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
t.train(ErrFunc::var, 10);
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
void saveFile(Str filen, Vec<double> ret)
{
    auto f = ofstream(filen);
    for(auto i : ret)
        f << i << endl;
    f.close();
}
TEST(T, 5)
{

    auto now = [](){return chrono::steady_clock::now();};
    //auto print_ret = [](auto K, auto name, auto diff, auto acc){cout << "K = " << K << " " << name << " spent " << chrono::duration <double, milli> (diff).count() << " ms, acc: " << acc << endl;};
    auto print_ret = [](auto K, auto name, auto diff, auto acc){cout <<  K << "," << name << "," << chrono::duration <double, milli> (diff).count() << "," << acc << endl;};
    //读取训练集数据
    //auto f = readFile("../data/Car_train.csv");
    //向量化
    //auto data = vectorizeData(f, mp);
    Eigen::Matrix data = readProject("../data/doc2vecTrainSet50D24000L.csv");
    Eigen::Matrix testdata = readTest("../data/doc2vecTestSet50D6000L.csv");
    for(int level = 8; level < 13; ++level)
    {
            //始始建树
            BaggingRegressTree t(data);
            //训练
            t.train(ErrFunc::var, 10, 40, level);
            //验证
            auto ret = t.predict(testdata);
            for(auto & i : ret)
            {
                if (i > 0.5)
                    i = 1;
                else
                    i = 0;
            }
            saveFile("16337237_4" + std::to_string(level-8) + ".csv",ret);
            cout << "File " << "16337237_4" + std::to_string(level-8) << " Saved" << endl;
    }
}
TEST(T, 4)
{

    auto now = [](){return chrono::steady_clock::now();};
    //auto print_ret = [](auto K, auto name, auto diff, auto acc){cout << "K = " << K << " " << name << " spent " << chrono::duration <double, milli> (diff).count() << " ms, acc: " << acc << endl;};
    auto print_ret = [](auto K, auto name, auto diff, auto acc){cout <<  K << "," << name << "," << chrono::duration <double, milli> (diff).count() << "," << acc << endl;};
    //读取训练集数据
    //auto f = readFile("../data/Car_train.csv");
    //向量化
    //auto data = vectorizeData(f, mp);
    Eigen::Matrix data = readProject("../data/doc2vecTrainSet50D24000L.csv");
    Eigen::Matrix testdata = readTest("../data/doc2vecTestSet50D6000L.csv");
    //K遍历2到8的值
    for(double K: range(5, 6))
    {
        int N = data.cols();
        int pieceSize = N / K;
        auto all = range(0, N);
        double acc = 0;
        auto start = now();
	for(int level = 6; level < 12; ++level)
	{
#pragma omp parallel for reduction(+: acc)
        for (int i = 0; i < K; ++i) {
            //分割验证集 1/K 的数据作为验证集
            auto vaildSetRange = range(i * pieceSize, (i + 1) * pieceSize);
            matrix_view<double> vaildSet(data, vaildSetRange);
            //其余数据作为训练集
            auto trainSetRange = view::set_difference(all, vaildSetRange);
            matrix_view<double> trainSet(data, trainSetRange);
            //始始建树
            BaggingRegressTree t(trainSet);
            //训练
            t.train(ErrFunc::var, 100, 40, level);
            //验证
            acc += (t.vaild(vaildSet));
        }
        acc /= K;
        auto diff = now() - start;
        cout << chrono::duration <double> (diff).count()  << "s" << endl;
        print_ret(level, "regress", diff, acc);

	}

    }
}
int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
