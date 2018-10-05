#include <range/v3/all.hpp>
#include <iostream>
using namespace ranges;
int main()
{
    auto colors = {"red", "green", "blue", "yellow"};
    for(const auto& [i, color] : view::zip(view::iota(0),colors))
    {
        std::cout << i << " " << color << std::endl;
    }
}