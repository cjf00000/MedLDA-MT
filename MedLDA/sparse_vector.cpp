//
// Created by Jianfei on 2019/3/2.
//
#include "sparse_vector.h"
#include <cmath>

void SparseVector::Update(int k, float delta) {
    bool found = false;
    for (auto &entry: data)
        if (entry.k == k) {
            found = true;
            entry.v += delta;
        }
    if (!found)
        data.push_back(Entry{k, delta});
}