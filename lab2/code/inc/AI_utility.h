//
// Created by 李新锐 on 03/10/2018.
//

#ifndef DECISIONTREE_AI_UTILITY_H
#define DECISIONTREE_AI_UTILITY_H

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "gtest/gtest.h"
#include "mkl.h"

using Matrix = Eigen::MatrixXd;

template<typename T>
using Vec = std::vector<T>;
using Str = std::string;
using FileData_t = Vec<Vec<Str>>;

FileData_t readFile(const Str& filen);

#endif //DECISIONTREE_AI_UTILITY_H
