//
// Created by Jianfei on 2019/2/27.
//

#ifndef MEDLDA_MEDLDA_H
#define MEDLDA_MEDLDA_H

#include "corpus.h"
#include "svm.h"
#include <vector>
#include <random>

class MedLDA {
public:
    MedLDA(Corpus &corpus, Corpus &testCorpus, int K, float alpha, float beta, float C, float ell, float eps);

    void ComputePhi();
    void SampleDoc(int d);
    void SampleTestDoc(int d);
    double SolveSVM();

    void Train();
    void Test();
    double Perplexity();

private:
    Corpus &corpus, &testCorpus;
    std::vector<SVM> svm;
    int K;
    float alpha, beta, C, ell;
    std::vector<int> cdk, cwk, ck;
    std::vector<float> phi, inv_ck;
    std::vector<float> prob;

    std::vector<float> test_cdk;

    std::mt19937 generator;
    std::uniform_real_distribution<float> u01;
};


#endif //MEDLDA_MEDLDA_H
