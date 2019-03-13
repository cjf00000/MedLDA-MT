//
// Created by Jianfei on 2019/3/2.
//

#ifndef MEDLDA_BIT_H
#define MEDLDA_BIT_H

#include <vector>

// Binary indexed tree
struct BIT {
    BIT();
    void Resize(int M);
    void Update(int k, float delta);
    void Build(float *data, int K);
    int GetIndex(float val);
    int LowBit(int x);
    float Sum();

    int N, M;
    std::vector<float> a;
};


#endif //MEDLDA_BIT_H
