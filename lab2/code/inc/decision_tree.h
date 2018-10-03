//
// Created by 李新锐 on 01/10/2018.
//

#ifndef DECISIONTREE_DECISION_TREE_H
#define DECISIONTREE_DECISION_TREE_H

#include "AI_utility.h"
#include <utility>
using std::pair;
using std::set;

class DecisionTree {
    const Matrix& trainX;
    const Matrix& trainY;
    struct Node {

    };
    Node* root;

public:
    DecisionTree(const Matrix& _trainX, const Matrix& _trainY)
            :trainX(_trainX), trainY(_trainY) {}
};

pair<Matrix, Matrix> vectorizeData(const FileData_t & fileData);

DecisionTree trainModel(const Matrix& trainX, const Matrix& trainY);


#endif //DECISIONTREE_DECISION_TREE_H
