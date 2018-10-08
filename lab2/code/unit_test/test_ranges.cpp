#include <range/v3/all.hpp>
#include <iostream>
#include "AI_utility.h"
using namespace ranges;

#include "gtest/gtest.h"

TEST(R, 1)
{
    auto colors = {"red", "green", "blue", "yellow"};
    for(const auto& [i, color] : view::zip(view::iota(0),colors))
    {
        std::cout << i << " " << color << std::endl;
    }
}

TEST(R, 2)
{
    Vec<int> v{view::concat(view::ints(0, 10), view::ints(20, 29))};
    for(auto i : v) cout << i << ", ";
}
int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

