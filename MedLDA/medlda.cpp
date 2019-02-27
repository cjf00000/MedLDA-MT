//
// Created by Jianfei on 2019/2/27.
//

#include "MedLDA.h"
#include <iostream>
using namespace std;

MedLDA::MedLDA(Corpus &corpus, int K, float alpha, float beta, float C, float ell)
    : corpus(corpus), K(K), alpha(alpha), beta(beta), C(C), ell(ell),
      cdk(corpus.num_data * K), cwk(corpus.V * K), ck(K),
      phi(corpus.V * K), inv_ck(K), prob(K)
{
    for (int d = 0; d < corpus.num_data; d++) {
        for (auto &token: corpus.w[d]) {
            token.z = generator() % K;
            cdk[d * K + token.z]++;
            if (isTrain(d)) {
                cwk[token.w * K + token.z]++;
                ck[token.z]++;
            }
        }
    }
    ComputePhi();

    cout << "Initialized" << endl;
}

void MedLDA::ComputePhi()
{
    for (int k = 0; k < K; k++)
        inv_ck[k] = 1.0 / (ck[k] + beta * corpus.V);
    for (int v = 0; v < corpus.V; v++)
        for (int k = 0; k < K; k++)
            phi[v * K + k] = (cwk[v * K + k] + beta) * inv_ck[k];
}

bool MedLDA::isTrain(int d)
{
    return d < corpus.num_train;
}

void MedLDA::SampleDoc(int d)
{
    auto *cd = cdk.data() + d * K;
    for (auto &token: corpus.w[d]) {
        auto *cw = cwk.data() + token.w * K;
        --cd[token.z];
        --cw[token.z];
        --ck[token.z];
        inv_ck[token.z] = 1.0 / (ck[token.z] + beta * corpus.V);

        float sum = 0;
        for (int k = 0; k < K; k++)
            prob[k] = sum += (cd[k] + alpha) * (cw[k] + beta) * inv_ck[k];

        float pos = sum * u01(generator);
        int k = 0;
        while (k + 1 < K && pos > prob[k]) k++;
        token.z = k;

        ++cd[token.z];
        ++cw[token.z];
        ++ck[token.z];
        inv_ck[token.z] = 1.0 / (ck[token.z] + beta * corpus.V);
    }
}

void MedLDA::Train()
{
    for (int iter = 0; iter < 100; iter++) {
        for (int d = 0; d < corpus.num_train; d++)
            SampleDoc(d);

        ComputePhi();
        double perplexity = Perplexity();
        cout << "Iteration " << iter << " perplexity " << perplexity << endl;
    }
}

double MedLDA::Perplexity()
{
    std::vector<float> theta(K);
    double log_likelihood = 0;
    for (int d = 0; d < corpus.num_train; d++) {
        for (int k = 0; k < K; k++)
            theta[k] = (cdk[d * K + k] + alpha) / (corpus.w[d].size() + alpha * K);
        for (auto &token: corpus.w[d]) {
            double l = 0;
            for (int k = 0; k < K; k++)
                l += theta[k] * phi[token.w * K + k];

            log_likelihood += log(l);
        }
    }
    return exp(-log_likelihood / corpus.train_T);
}
