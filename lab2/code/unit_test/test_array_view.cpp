//
// Created by 李新锐 on 04/10/2018.
//

#include <gtest/gtest.h>
#include "array_view.h"
#include "iostream"
using namespace std;

TEST(A, 5)
{
    std::vector<std::vector<int>> trainX = {
            {1, 2, 3, 4},
            {2, 3, 4, 5},
            {3, 4, 5, 6}
    };
    std::set<size_t> D = {1,2};
    array_view<int> v(trainX, D);
    for(auto i : v)
    {
        for(size_t j = 0; j < i.size(); ++j)
        {
            cout  << j << ',';
        }
        cout << endl;
    }
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
