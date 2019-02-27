//
// Created by Jianfei on 2019/2/27.
//

#ifndef MEDLDA_MEDLDA_H
#define MEDLDA_MEDLDA_H

#include "corpus.h"
#include <vector>
#include <random>

class MedLDA {
public:
    MedLDA(Corpus &corpus, int K, float alpha, float beta, float C, float ell);

    void ComputePhi();

    bool isTrain(int d);

    void SampleDoc(int d);

    void Train();

    double Perplexity();

private:
    Corpus& corpus;
    int K;
    float alpha, beta, C, ell;
    std::vector<int> cdk, cwk, ck;
    std::vector<float> phi, inv_ck;
    std::vector<float> prob;
    std::mt19937 generator;
    std::uniform_real_distribution<float> u01;
};


#endif //MEDLDA_MEDLDA_H
