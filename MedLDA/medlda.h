//
// Created by Jianfei on 2019/2/27.
//

#ifndef MEDLDA_MEDLDA_H
#define MEDLDA_MEDLDA_H

#include "corpus.h"
#include "svm.h"
#include "gflags/gflags.h"
#include "sparse_vector.h"
#include "bit.h"
#include <vector>
#include <random>

DECLARE_bool(fast_sampling);
DECLARE_bool(fast_precompute);
DECLARE_double(epsilon);

class MedLDA {
public:
    MedLDA(Corpus &corpus, Corpus &testCorpus, int K, float alpha, float beta, float C, float ell, float eps);

    void ComputePhi();
    void ComputeDocProb();
    void SampleDoc(int d);
    void SampleWord(int w);
    void SampleTestDoc(int d);
    void SolveSVM();

    void Train();
    void Test();
    double Perplexity();

private:
    Corpus &corpus, &testCorpus;
    std::vector<SVM> svm;
    int K, nSV, nSVMIters;
    int num_reject, doc_prob_nnz, cdk_nnz;
    int num_1, num_2, num_3;
    float alpha, beta, C, ell;
    std::vector<int> cdk, cwk, ck;
    std::vector<SparseVector> sparse_cdk;
    std::vector<float> phi, inv_ck;
    std::vector<float> prob, doc_prob;
    std::vector<SparseVector> doc_prob_hat;

    std::vector<float> test_cdk;

    std::mt19937 generator;
    std::uniform_real_distribution<float> u01;

    double svmTime, classTime, ldaTime, avg_cdk;
    float accuracy, micro_f1, macro_f1;

    BIT bit;
    int global_iter;
};


#endif //MEDLDA_MEDLDA_H
