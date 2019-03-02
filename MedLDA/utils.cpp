//
// Created by Jianfei on 2019/3/1.
//
#include <cmath>
#include <algorithm>
#include "utils.h"
using namespace std;

void Softmax(float *a, int N)
{
    float max_p = -1e9;
    for (int i = 0; i < N; i++) max_p = max(max_p, a[i]);
    float sum = 0;
    for (int i = 0; i < N; i++) sum += a[i] = exp(a[i] - max_p);
    sum = 1.0 / sum;
    for (int i = 0; i < N; i++) a[i] *= sum;
}
