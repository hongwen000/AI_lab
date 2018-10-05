//
// Created by 李新锐 on 05/10/2018.
//

#include "gtest/gtest.h"
#include "AI_utility.h"
TEST(A, 1)
{
    readFile("../data/Car_train.csv");
}
int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

