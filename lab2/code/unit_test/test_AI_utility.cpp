//
// Created by 李新锐 on 05/10/2018.
//

#include "gtest/gtest.h"
#include "AI_utility.h"
TEST(A, 1)
{
    Eigen::MatrixXd ret = readProject("../data/doc2vecTrainSet50D24000L.csv");
    cout << ret.col(0) << endl;
}
int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

