//
// Created by 李新锐 on 04/10/2018.
//

#include <fstream>
#include "gtest/gtest.h"
#include "iostream"
#include "decision_tree.h"
#include <chrono>
using namespace std;

TEST(T, 0)
{
    Matrix data(5, 14);
    data <<0,0,1,2,2,2,1,0,0,2,0,1,1,2,
            2,2,2,1,0,0,0,1,0,1,1,1,2,1,
            0,0,0,0,1,1,1,0,1,1,1,0,1,0,
            0,1,0,0,0,1,1,0,0,0,1,1,0,1,
            0,0,1,1,1,0,1,0,1,1,1,1,1,0;
    constexpr double BLOCK_NUM = 5.0;
    auto all = view::ints(0, (int)data.cols());
    for(int i = 0; i < BLOCK_NUM; ++i)
    {
        auto vaildSetRange = view::ints((int)(i / BLOCK_NUM * data.cols()), (int)((i + 1) / BLOCK_NUM * data.cols()));
        auto trainSetRange = view::set_difference(all, vaildSetRange);
        matrix_view<int> vaildSet(data, vaildSetRange);
        matrix_view<int> trainSet(data, trainSetRange);
    }
}
TEST(T, 1)
{
    Matrix a(5, 14);
    a <<0,0,1,2,2,2,1,0,0,2,0,1,1,2,
        2,2,2,1,0,0,0,1,0,1,1,1,2,1,
        0,0,0,0,1,1,1,0,1,1,1,0,1,0,
        0,1,0,0,0,1,1,0,0,0,1,1,0,1,
        0,0,1,1,1,0,1,0,1,1,1,1,1,0;
    DecisionTree t(a, {3,3,2,2});
    JudgeFunc_t func = JudgeFunc::CART;
    t.train(func);
    std::map<int, std::vector<std::string>> mp = {
            {0, {"年龄", "<= 30", "31...40", ">40"}},
            {1, {"收入", "low", "medium", "high"}},
            {2, {"学生?", "no", "yes"}},
            {3, {"信用等级", "fair", "excellent"}},
            {-1, {"no", "yes"}}
    };

    auto ret = t.predict(a);
    for(auto i : ret)
        cout << i << ",";
    cout << endl;
    cout << t.vaild(a);
    auto s = t.print(mp);
    std::cout << "--------" << std::endl;
    std::cout << s << std::endl;
    std::ofstream f("/Users/lixinrui/1.dot");
    f << s;
    f.close();
}

TEST(T, 2)
{
    auto f = readFile("../data/Car_train.csv");
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
                    {"£ø", 0}
            }
    };
    Matrix trainSetOri = vectorizeData(f, mp);
    auto trainSet = matrix_view<int>(trainSetOri, view::ints(0, 31));
    DecisionTree t(trainSet, {4,4,4,3,3,3});
    JudgeFunc_t func = JudgeFunc::CART;
    auto start = chrono::steady_clock::now();
    t.train(func);
    auto end = chrono::steady_clock::now();
    auto diff = end - start;
    cout << chrono::duration <double, milli> (diff).count() << " ms" << endl;
    std::map<int, std::vector<std::string>> rmp = {
            {0,  {"buying",   "low",   "med", "high", "v-high"}},
            {1,  {"maint",    "low",   "med", "high", "v-high"}},
            {2,  {"doors",    "2",     "3",   "4",    "5", "5-more"}},
            {3,  {"persons",  "2",     "4",   "more"}},
            {4,  {"lug_boot", "small", "med", "big"}},
            {5,  {"safety",   "low",   "med", "high"}},
            {-1, {"0",      "1"}}
    };
    auto s = t.print(rmp);
    std::ofstream of("/Users/lixinrui/1.dot");
    of << s;
    of.close();
    cout << t.vaild(trainSet);
}
int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
