//
// Created by Jianfei on 2019/3/2.
//

#include "bit.h"
#include <algorithm>
using namespace std;

BIT::BIT() {}

void BIT::Resize(int M) {
    this->M = N = M;
    while (N != LowBit(N)) N += LowBit(N);
    a.resize(N+1);
    fill(a.begin(), a.end(), 0);
}

void BIT::Update(int k, float delta)
{
    k++;
    do {
        a[k] += delta;
        k += LowBit(k);
    } while (k <= N);
}

void BIT::Build(float *data, int K) {
    for (int i = 0; i < K; i++)
        a[i+1] = data[i];

    for (int i = 1; i <= N; i++) {
        int j = i - 1;
        int child = 1;
        while ((j & child) == child) {
            a[i] += a[i - child];
            child *= 2;
        }
    }
}

int BIT::GetIndex(float val)
{
    int pos = 0;
    int step = N / 2;
    while (step) {
        if (val > a[pos + step]) {
            val -= a[pos + step];
            pos += step;
        }
        step /= 2;
    }
    pos = min(pos, M-1);
    return pos;
}

int BIT::LowBit(int x) {
    return x & (-x);
}

float BIT::Sum() {
    return a.back();
}