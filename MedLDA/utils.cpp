//
// Created by Jianfei on 2019/3/1.
//
#include <cmath>
#include <algorithm>
#include "utils.h"
using namespace std;

void Softmax(std::vector<double> &p)
{
    double max_p = -1e9;
    for (auto &pp: p) max_p = max(max_p, pp);
    double sum = 0;
    for (auto &pp: p) sum += pp = exp(pp - max_p);
    sum = 1.0 / sum;
    for (auto &pp: p) pp *= sum;
}
