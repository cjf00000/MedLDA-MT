//
// Created by Jianfei on 2019/3/2.
//
#include "sparse_vector.h"
#include <cmath>

void SparseVector::From(std::vector<float> &dense) {
    data.clear();
    for (size_t i = 0; i < dense.size(); i++)
        if (fabs(dense[i]) > 1e-7)
            data.push_back(Entry{(int)i, dense[i]});
}