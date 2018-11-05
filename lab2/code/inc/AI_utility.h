//
// Created by 李新锐 on 03/10/2018.
//

#ifndef DECISIONTREE_AI_UTILITY_H
#define DECISIONTREE_AI_UTILITY_H

#include <iostream>
using std::cout;
using std::endl;
//#include <Eigen/Dense>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "gtest/gtest.h"
#ifdef USE_MLK
    #include "mkl.h"
#endif


template<typename T>
using Vec = std::vector<T>;
using Str = std::string;
using FileData_t = Vec<Vec<Str>>;
using Matrix = Eigen::MatrixXi;
using Vector = Eigen::VectorXi;
using Eigen::ArrayXi;

FileData_t readFile(const Str& filen);

template <typename Derived>
bool all_equal(const Eigen::DenseBase<Derived>& b)
{
    if(b.size() == 0)
        return true;
    auto v = b[0];
    for(Eigen::Index i = 0; i < b.cols(); ++i)
    {
        for(Eigen::Index j = 0; j < b.rows(); ++j)
        {
            if(v != b(j, i))
                return false;
        }
    }
    return true;
}

template <typename Derived>
bool all_equal(const Eigen::DenseBase<Derived>& b, int v)
{
    for(Eigen::Index i = 0; i < b.cols(); ++i)
    {
        for(Eigen::Index j = 0; j < b.rows(); ++j)
        {
            if(v != b(j, i))
                return false;
        }
    }
    return true;
}
Eigen::MatrixXd readProject(const Str& filen);
Eigen::MatrixXd readTest(const Str& filen);
#endif //DECISIONTREE_AI_UTILITY_H
