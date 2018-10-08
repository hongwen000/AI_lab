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
                {"£ø", 0}
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
    auto f = readFile("data/Car_train.csv");
    auto data = vectorizeData(f, mp);
    constexpr double BLOCK_NUM = 5.0;
    auto all = view::ints(0, (int)data.cols());
    for(int i = 0; i < BLOCK_NUM; ++i)
    {
        auto vaildSetRange = view::ints((int)(i / BLOCK_NUM * data.cols()), (int)((i + 1) / BLOCK_NUM * data.cols()));
        auto trainSetRange = view::set_difference(all, vaildSetRange);
        matrix_view<int> vaildSet(data, vaildSetRange);
        matrix_view<int> trainSet(data, trainSetRange);
        for(auto& [name, Func] : JudgeFuncs)
        {
            DecisionTree t(trainSet, {4,4,4,3,3,3});
            auto start = chrono::steady_clock::now();
            t.train(Func);
            auto end = chrono::steady_clock::now();
            auto diff = end - start;
            auto acc = t.vaild(vaildSet);
            cout << name << " spent " << chrono::duration <double, milli> (diff).count() << " ms, acc: " << acc << endl;
        }
    }
}