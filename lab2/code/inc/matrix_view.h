//
// Created by 李新锐 on 04/10/2018.
//

#ifndef DECISIONTREE_MATRIX_VIEW_H
#define DECISIONTREE_MATRIX_VIEW_H

#include "cmath"
#include "vector"
#include "string"
#include "functional"
#include "set"
#include "iostream"
#include <eigen3/Eigen/Dense>
#include "range/v3/view.hpp"
#include <memory>

template <typename T>
class matrix_view {
    const Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> vec;
    // const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& vec;
    std::vector<Eigen::Index> view;
public:
    matrix_view(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& _vec):vec(_vec.data(), _vec.rows(), _vec.cols())
    {
        for(Eigen::Index i = 0; i < vec.cols(); ++i)
        {
            view.push_back(i);
        }
//        for(auto i : view)
//        {
//            std::cout << i << ',';
//        }
//        std::cout << std::endl;
    }
    matrix_view(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& _vec, const std::vector<Eigen::Index>& ilist): vec(_vec.data(), _vec.rows(), _vec.cols())
    {
        for(const auto& i: ilist)
        {
            if(i < _vec.cols())
                view.push_back(i);
            else
                throw(std::out_of_range("matrix_view ~ ctor ~ vec.cols = " + std::to_string(_vec.cols()) + " ~ i = " + std::to_string(i)));
        }
    }
    matrix_view(const matrix_view<T>& father, const std::vector<Eigen::Index>& ilist): vec(father.vec.data(), father.vec.rows(), father.vec.cols())
    {
        for(const auto& i: ilist)
        {
            if(i < father.cols())
                view.push_back(i);
            else
                throw(std::out_of_range("matrix_view ~ ctor ~ father.cols = " + std::to_string(father.cols()) + " ~ i = " + std::to_string(i)));
        }

    }

    long cols() const
    {
        return view.size();
    }

    long rows() const
    {
        return vec.rows();
    }

    long size() const
    {
        return view.size() * vec.rows();
    }

    bool empty() const
    {
        return size() == 0;
    }

    const T operator()(Eigen::Index i, Eigen::Index j) const
    {
        if(std::abs(i) >= rows())
        {
            throw(std::out_of_range("matrix_view ~ op() ~ i = " + std::to_string(rows()) + " ~ i = " + std::to_string(i)));
        }
        if(i < 0)
            i = rows() + i;

        if(std::abs(j) >= cols())
        {
            throw(std::out_of_range("matrix_view ~ op() ~ view.size = " + std::to_string(view.size()) + " ~ j = " + std::to_string(j)));
        }
        if(j < 0)
            j = cols() + j;
        return vec(i, view[j]);
    }

    Eigen::Matrix<T, 1, Eigen::Dynamic> row(Eigen::Index r) const
    {
//        std::cout << "[DEBUG in row()] " << std::endl;
//        std::cout << vec << std::endl;
        if(std::abs(r) >= rows())
        {
            throw(std::out_of_range("matrix_view ~ row() ~ rows = " + std::to_string(rows()) + " ~ r = " + std::to_string(r)));
        }
//        std::cout << "rows() = " << rows() << std::endl;
        if(r < 0)
            r = rows() + r;
        Eigen::Matrix<T, 1, Eigen::Dynamic> ret(cols());
        for(Eigen::Index i = 0; i < cols(); ++i)
        {
            ret(0, i) = vec(r, view[i]);
        }
        return ret;
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> col(Eigen::Index c) const
    {
        if(std::abs(c) >= cols())
        {
            throw(std::out_of_range("matrix_view ~ col() ~ cols = " + std::to_string(cols()) + " ~ c = " + std::to_string(c)));
        }
        if(c < 0)
            c = cols() + c;
        Eigen::Matrix<T, Eigen::Dynamic, 1> ret(rows());
        for(Eigen::Index i = 0; i < rows(); ++i)
        {
            ret(i, 0) = vec(i, view[c]);
        }
        return ret;
    }
};
#endif //DECISIONTREE_MATRIX_VIEW_H
