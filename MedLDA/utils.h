//
// Created by Jianfei on 2019/3/1.
//

#ifndef MEDLDA_UTILS_H
#define MEDLDA_UTILS_H

#include <vector>

struct Entry {
    int k;
    float v;
};

void Softmax(float *a, int N);

#endif //MEDLDA_UTILS_H
