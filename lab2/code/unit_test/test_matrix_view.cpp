//
// Created by 李新锐 on 04/10/2018.
//

#include "matrix_view.h"
#include "gtest/gtest.h"
#include "iostream"
using namespace std;
TEST(M, 6)
{
    Eigen::MatrixXi a(2, 4);
    a << 1,2,5,7,
         3,4,6,8;
    matrix_view<int> v(a, {1,3});
    v.select_row({0});
    for(long i = 0; i < v.rows(); ++i)
    {
        for(long j = 0; j < v.cols(); ++j)
        {
            cout << v(i,j) << ',';
        }
    cout << endl;
    }
    cout << "---------" << endl;
    cout << v.row(-1).sum() << endl;
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
