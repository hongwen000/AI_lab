//
// Created by 李新锐 on 01/10/2018.
//

#include <iostream>
#include <Eigen/Dense>
#include "gtest/gtest.h"
#include "mkl.h"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;


using namespace std;
using namespace Eigen;
using Eigen::MatrixXd;

class Foo
{
public:
    Eigen::Vector2d v;
};

TEST(E, 1)
{
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;
}

TEST(E, 2)
{
    MatrixXd m(2,2);
    m(0,0) = 1;
    m(1,0) = 2;
    m(0,1) = 3;
    m(1,1) = 4;
    m = (m + MatrixXd::Constant(2, 2, 1)) * 50;
    cout << "m is " << endl << m << endl;
    VectorXd v(2);
    v << 1, 2;
    cout << v << endl;
    cout << "m * v = " << endl << m * v << endl;

}

TEST(E, 3)
{
    MatrixXd m(2, 5);
    m.resize(4, 3);
    m.transposeInPlace();
}

TEST(E, 4)
{
    Foo *foo = new Foo;
    foo-> v << 1, 2;
    cout << foo->v << endl;
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

