//
// Created by Jianfei on 2019/2/27.
//

#include "medlda.h"
#include "utils.h"
#include <iostream>
using namespace std;

MedLDA::MedLDA(Corpus &corpus, Corpus &testCorpus,
               int K, float alpha, float beta, float C, float ell, float eps)
    : corpus(corpus), testCorpus(testCorpus),
      K(K), alpha(alpha), beta(beta), C(C), ell(ell),
      cdk(corpus.num_docs * K), cwk(corpus.V * K), ck(K),
      phi(corpus.V * K), inv_ck(K), prob(K),
      test_cdk(testCorpus.num_docs * K)
{
    for (int c = 0; c < corpus.num_classes; c++)
        svm.push_back(SVM(corpus.num_docs, K, 2 * C, ell, eps));

    for (int d = 0; d < corpus.num_docs; d++) {
        for (auto &token: corpus.w[d]) {
            token.z = generator() % K;
            cdk[d * K + token.z]++;
            cwk[token.w * K + token.z]++;
            ck[token.z]++;
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

void MedLDA::SampleDoc(int d)
{
    auto *cd = cdk.data() + d * K;
    std::vector<double> doc_prob(K);
    for (int c = 0; c < corpus.num_classes; c++)
        for (int k = 0; k < K; k++)
            doc_prob[k] += svm[c].w[k] * svm[c].alpha[d] * corpus.ys[c][d];
    Softmax(doc_prob);

    for (auto &token: corpus.w[d]) {
        auto *cw = cwk.data() + token.w * K;
        --cd[token.z];
        --cw[token.z];
        --ck[token.z];
        inv_ck[token.z] = 1.0 / (ck[token.z] + beta * corpus.V);

        float sum = 0;
        for (int k = 0; k < K; k++)
            prob[k] = sum += (cd[k] + alpha) * (cw[k] + beta) * inv_ck[k] * doc_prob[k];

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

void MedLDA::SampleTestDoc(int d)
{
    vector<int> cd(K);
    auto *mean_cd = test_cdk.data() + d * K;
    fill(mean_cd, mean_cd + K, 0);
    for (auto &token: testCorpus.w[d]) {
        token.z = generator() % K;
        cd[token.z]++;
    }

    for (int iter = 0; iter <= 20; iter++) {
        for (auto &token: testCorpus.w[d]) {
            auto *wp = phi.data() + token.w * K;
            --cd[token.z];

            float sum = 0;
            for (int k = 0; k < K; k++)
                prob[k] = sum += (cd[k] + alpha) * wp[k];

            float pos = sum * u01(generator);
            int k = 0;
            while (k + 1 < K && pos > prob[k]) k++;
            token.z = k;

            ++cd[token.z];
        }
        if (iter >= 10)
            for (int k = 0; k < K; k++)
                mean_cd[k] += (double)cd[k] / 10 / testCorpus.w[d].size();
    }
}

double MedLDA::SolveSVM()
{
    std::vector<Feature> X;
    for (int d = 0; d < corpus.num_docs; d++) {
        Feature doc;
        for (int k = 0; k < K; k++)
            if (cdk[d * K + k])
                doc.push_back(Entry{k, (float)cdk[d * K + k] / corpus.w[d].size()});
        X.push_back(move(doc));
    }
    vector<vector<double>> pred;
    nSV = 0;
    for (int c = 0; c < corpus.num_classes; c++) {
        svm[c].Solve(X, corpus.ys[c]);
        pred.push_back(move(svm[c].Predict(X)));
        nSV += svm[c].nSV();
    }
    return corpus.Accuracy(pred);
}

void MedLDA::Train()
{
    for (int iter = 0; iter < 100; iter++) {
        double acc = SolveSVM();

        for (int d = 0; d < corpus.num_docs; d++)
            SampleDoc(d);

        ComputePhi();
        double perplexity = Perplexity();
        cout << "Iteration " << iter
             << " perplexity " << perplexity
             << " nSV " << nSV
             << " training accuracy " << acc << endl;

        if (iter % 10 == 0)
            Test();
    }
}

void MedLDA::Test()
{
    std::vector<Feature> X;
    for (int d = 0; d < testCorpus.num_docs; d++) {
        Feature doc;
        SampleTestDoc(d);
        for (int k = 0; k < K; k++)
            doc.push_back(Entry{k, (float)test_cdk[d * K + k]});
        X.push_back(move(doc));
    }
    vector<vector<double>> pred;
    for (int c = 0; c < corpus.num_classes; c++)
        pred.push_back(move(svm[c].Predict(X)));

    cout << "Testing accuracy = " << testCorpus.Accuracy(pred) << endl;
}

double MedLDA::Perplexity()
{
    std::vector<float> theta(K);
    double log_likelihood = 0;
    for (int d = 0; d < corpus.num_docs; d++) {
        for (int k = 0; k < K; k++)
            theta[k] = (cdk[d * K + k] + alpha) / (corpus.w[d].size() + alpha * K);
        for (auto &token: corpus.w[d]) {
            double l = 0;
            for (int k = 0; k < K; k++)
                l += theta[k] * phi[token.w * K + k];

            log_likelihood += log(l);
        }
    }
    return exp(-log_likelihood / corpus.T);
}
