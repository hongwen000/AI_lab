#include "decision_tree.h"
#include "chrono"
#include "algorithm"
using namespace std;

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
std::map<int, std::vector<std::string>> rmp = {
        {0,  {"buying",   "low",   "med", "high", "v-high"}},
        {1,  {"maint",    "low",   "med", "high", "v-high"}},
        {2,  {"doors",    "2",     "3",   "4",    "5", "5-more"}},
        {3,  {"persons",  "2",     "4",   "more"}},
        {4,  {"lug_boot", "small", "med", "big"}},
        {5,  {"safety",   "low",   "med", "high"}},
        {-1, {"0",      "1"}}
};

std::map<std::string, JudgeFunc_t> JudgeFuncs = {
        {"ID3", JudgeFunc::ID3},
        {"C45", JudgeFunc::C45},
        {"CART", JudgeFunc::CART}
};

int main()
{
    auto now = [](){return chrono::steady_clock::now();};
    //auto print_ret = [](auto K, auto name, auto diff, auto acc){cout << "K = " << K << " " << name << " spent " << chrono::duration <double, milli> (diff).count() << " ms, acc: " << acc << endl;};
    auto print_ret = [](auto K, auto name, auto diff, auto acc){cout <<  K << "," << name << "," << chrono::duration <double, milli> (diff).count() << "," << acc << endl;};
    //读取训练集数据
    auto f = readFile("data/Car_train.csv");
    //向量化
    auto data = vectorizeData(f, mp);
    //K遍历2到8的值
    for(double K: range(2, 9))
    {
        int  N = data.cols();
        int  pieceSize = N / K;
        auto all = range(0, N);
        double diff = 0;
        double acc = 0;
        string name = "CART";
        auto Func = JudgeFunc::CART;
        for(auto i : range(0, K))
        {
            //分割验证集 1/K 的数据作为验证集
            auto vaildSetRange = range(i * pieceSize, (i + 1) * pieceSize);
            matrix_view<int> vaildSet(data, vaildSetRange);
            //其余数据作为训练集
            auto trainSetRange = view::set_difference(all, vaildSetRange);
            matrix_view<int> trainSet(data, trainSetRange);
            //遍历三种决策树方法
            //for(auto& [name, Func] : JudgeFuncs)
            {
                auto featureValRanges = {4,4,4,3,3,3};
                DecisionTree t(trainSet, featureValRanges);
                //计时并建树
                auto start = now();
                t.train(Func);
                //t.prune();
                auto diff = now() - start;
                //验证并打印结果
                acc += t.vaild(vaildSet);
            }
        }
        acc /= K;
        print_ret(K, name, diff, acc);
    }
    {
        matrix_view<int> trainSet(data);
        auto featureValRanges = {4,4,4,3,3,3};
        DecisionTree t(trainSet, featureValRanges);
        t.train(JudgeFunc::C45);
        auto f = readFile("data/Car_test.csv");
        auto data = vectorizeData(f, mp);
        auto ret =t.predict(data);
        for(auto i : ret)
        {
            cout << i << endl;
        }
        cout << ranges::v3::accumulate(ret, 0) << endl;
    }

}