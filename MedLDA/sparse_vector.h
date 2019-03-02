//
// Created by Jianfei on 2019/3/2.
//

#ifndef MEDLDA_SPARSE_VECTOR_H
#define MEDLDA_SPARSE_VECTOR_H

#include "utils.h"
#include <vector>

struct SparseVector {
    void From(std::vector<float> &dense);
    int Size() { return data.size(); }

    std::vector<Entry> data;
};

#endif //MEDLDA_SPARSE_VECTOR_H
