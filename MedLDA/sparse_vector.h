//
// Created by Jianfei on 2019/3/2.
//

#ifndef MEDLDA_SPARSE_VECTOR_H
#define MEDLDA_SPARSE_VECTOR_H

#include "utils.h"
#include <vector>

struct SparseVector {
    template <class T>
    void From(T *dense, int N) {
        data.clear();
        for (int i = 0; i < N; i++)
            if (fabs(dense[i]) > 1e-7)
                data.push_back(Entry{(int)i, dense[i]});
    }

    int Size() { return data.size(); }
    void Update(int k, float delta);

    std::vector<Entry> data;
};

#endif //MEDLDA_SPARSE_VECTOR_H
