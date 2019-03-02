//
// Created by Jianfei on 2019/2/27.
//

#include "medlda.h"
#include "utils.h"
#include "clock.h"
#include "bit.h"
#include <iostream>
using namespace std;

DEFINE_bool(fast_sampling, false, "Fast sampling of topic assignment");
DEFINE_bool(fast_precompute, false, "Sparsity-aware pre-compute of discriminative weight.");
DEFINE_double(epsilon, 0.01, "Lower bound of doc prob");

MedLDA::MedLDA(Corpus &corpus, Corpus &testCorpus,
               int K, float alpha, float beta, float C, float ell, float eps)
    : corpus(corpus), testCorpus(testCorpus),
      K(K), alpha(alpha), beta(beta), C(C), ell(ell),
      cdk(corpus.num_docs * K), cwk(corpus.V * K), ck(K), sparse_cdk(corpus.num_docs),
      phi(corpus.V * K), inv_ck(K), prob(K), doc_prob(corpus.num_docs * K),
      doc_prob_hat(corpus.num_docs),
      test_cdk(testCorpus.num_docs * K)
{
    for (int c = 0; c < corpus.num_classes; c++)
        svm.push_back(SVM(corpus.num_docs, K, 2 * C, ell, eps));

    if (!FLAGS_fast_sampling) {
        corpus.AllocZDoc(K);
        for (int d = 0; d < corpus.num_docs; d++) {
            corpus.ForDoc(d, [&](int w, int k) {
                cdk[d * K + k]++;
                cwk[w * K + k]++;
                ck[k]++;
            });
        }
    } else {
        corpus.AllocZWord(K);
        for (int w = 0; w < corpus.V; w++) {
            corpus.ForWord(w, [&](int d, int k) {
                cdk[d * K + k]++;
                cwk[w * K + k]++;
                ck[k]++;
            });
        }
    }

    testCorpus.AllocZDoc(K);
    ComputePhi();

    cout << "Initialized" << endl;
    if (FLAGS_fast_sampling)
        cout << "Fast sampling..." << endl;
}

void MedLDA::ComputePhi()
{
    for (int k = 0; k < K; k++)
        inv_ck[k] = 1.0 / (ck[k] + beta * corpus.V);
    for (int v = 0; v < corpus.V; v++)
        for (int k = 0; k < K; k++)
            phi[v * K + k] = (cwk[v * K + k] + beta) * inv_ck[k];
}

void MedLDA::ComputeDocProb()
{
    doc_prob_nnz = 0;
    Clock clk;
    for (int d = 0; d < corpus.num_docs; d++) {
        auto *dp = doc_prob.data() + d * K;
        for (int c = 0; c < corpus.num_classes; c++) {
            if (!FLAGS_fast_precompute || svm[c].alpha[d] > 1e-7)
                for (int k = 0; k < K; k++)
                    dp[k] += svm[c].w[k] * svm[c].alpha[d] * corpus.ys[c][d];
        }

        Softmax(dp, K);

        // Sparsify
        auto &dph = doc_prob_hat[d];
        dph.data.clear();
        for (int k = 0; k < K; k++)
            if (dp[k] > FLAGS_epsilon) {
                dph.data.push_back(Entry{k, dp[k] - FLAGS_epsilon});
                doc_prob_nnz++;
            }

//        if (u01(generator) < 0.0003) {
//            float ss = 0;
//            for (auto &entry: dph.data)
//                cout << entry.k << ':' << entry.v << ' ';
//            cout << endl;
//            for (int c = 0; c < corpus.num_classes; c++) {
//                cout << corpus.ys[c][d] << ' ' << svm[c].alpha[d] << endl;
//            }
//        }
    }
    classTime += clk.toc();
}

void MedLDA::SampleDoc(int d)
{
    auto *cd = cdk.data() + d * K;
    auto *dp = doc_prob.data() + d * K;
    Clock clk;
    corpus.ForDoc(d, [&](int w, int &z) {
        auto *cw = cwk.data() + w * K;
        --cd[z];
        --cw[z];
        --ck[z];
        inv_ck[z] = 1.0 / (ck[z] + beta * corpus.V);

        float sum = 0;
        for (int k = 0; k < K; k++)
            prob[k] = sum += (cd[k] + alpha) * (cw[k] + beta) * inv_ck[k] * dp[k];

        float pos = sum * u01(generator);
        int k = 0;
        while (k + 1 < K && pos > prob[k]) k++;
        z = k;

        ++cd[z];
        ++cw[z];
        ++ck[z];
        inv_ck[z] = 1.0 / (ck[z] + beta * corpus.V);
    });
    ldaTime += clk.toc();
}

void MedLDA::SampleWord(int w)
{
    Clock clk;
    auto *cw = cwk.data() + w * K;

    std::vector<float> phi(K);
    //float phi_sum = 0;
    BIT bit;
    bit.Resize(K);
    for (int k = 0; k < K; k++) {
        phi[k] = (cw[k] + beta) / (ck[k] + beta * corpus.V);
        bit.Update(k, phi[k]);
    }

    std::vector<float> prob_1(K), prob_2(K);
    corpus.ForWord(w, [&](int d, int &z) {
        auto *cd = cdk.data() + d * K;
        auto &s_cd = sparse_cdk[d];
        auto *dp = doc_prob.data() + d * K;
        auto &dph = doc_prob_hat[d];
        --cd[z];
        s_cd.Update(z, -1);
        --cw[z];
        --ck[z];
        float old_phi = phi[z];
        float new_phi = (cw[z] + beta) / (ck[z] + beta * corpus.V);
        bit.Update(z, new_phi - old_phi);
        phi[z] = new_phi;

        int k;
        while (1) {
            float sum_1 = 0;
            float sum_2 = 0;
            float sum_3 = alpha * FLAGS_epsilon * bit.Sum();
            for (size_t idx = 0; idx < s_cd.Size(); idx++) {
                auto &entry = s_cd.data[idx];
                prob_1[idx] = sum_1 += entry.v * phi[entry.k] * dp[entry.k];
            }

            for (size_t idx = 0; idx < dph.Size(); idx++) {
                auto &entry = dph.data[idx];
                prob_2[idx] = sum_2 += alpha * phi[entry.k] * entry.v;
            }
            float pos = (sum_1 + sum_2 + sum_3) * u01(generator);
            k = 0;
            if (pos < sum_1) {
                num_1++;
                while (k + 1 < s_cd.Size() && pos > prob_1[k]) k++;
                if (k >= s_cd.Size()) {
                    // Shouldn't happen, just for numerical safety
                    num_reject++;
                    continue;
                }
                k = s_cd.data[k].k;
                break;
            } else if (pos < sum_1 + sum_2) {
                num_2++;
                pos -= sum_1;
                while (k + 1 < dph.Size() && pos > prob_2[k]) k++;
                if (k >= dph.Size()) {
                    // Shouldn't happen, just for numerical safety
                    num_reject++;
                    continue;
                }
                k = dph.data[k].k;
                break;
            } else {
                num_3++;
                float new_pos = u01(generator) * bit.Sum();
                k = bit.GetIndex(new_pos);

                float gap = max(FLAGS_epsilon - dp[k], 0.0);
                if (u01(generator) * FLAGS_epsilon < gap)
                    num_reject++;
                else
                    break;
            }
        }
        z = k;

        ++cd[z];
        s_cd.Update(z, 1);
        ++cw[z];
        ++ck[z];
        old_phi = phi[z];
        new_phi = (cw[z] + beta) / (ck[z] + beta * corpus.V);
        bit.Update(z, new_phi - old_phi);
        phi[z] = new_phi;
    });
    ldaTime += clk.toc();
}

void MedLDA::SampleTestDoc(int d)
{
    vector<int> cd(K);
    auto *mean_cd = test_cdk.data() + d * K;
    fill(mean_cd, mean_cd + K, 0);
    testCorpus.ForDoc(d, [&](int w, int z) { cd[z]++; });

    for (int iter = 0; iter <= 20; iter++) {
        testCorpus.ForDoc(d, [&](int w, int &z) {
            auto *wp = phi.data() + w * K;
            --cd[z];

            float sum = 0;
            for (int k = 0; k < K; k++)
                prob[k] = sum += (cd[k] + alpha) * wp[k];

            float pos = sum * u01(generator);
            int k = 0;
            while (k + 1 < K && pos > prob[k]) k++;
            z = k;

            ++cd[z];
        });
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
    nSVMIters = 0;
    for (int c = 0; c < corpus.num_classes; c++) {
        svm[c].Solve(X, corpus.ys[c]);
        pred.push_back(move(svm[c].Predict(X)));
        nSV += svm[c].nSV();
        nSVMIters += svm[c].num_iters;
    }
    return corpus.Accuracy(pred);
}

void MedLDA::Train()
{
    for (int iter = 0; iter < 100; iter++) {
        classTime = ldaTime = cdk_nnz = 0;
        num_1 = num_2 = num_3 = 0;

        Clock clk;
        double acc = SolveSVM();
        svmTime = clk.toc();

        ComputeDocProb();
        num_reject = 0;
        if (!FLAGS_fast_sampling) {
            for (int d = 0; d < corpus.num_docs; d++)
                SampleDoc(d);
        } else {
            for (int d = 0; d < corpus.num_docs; d++) {
                sparse_cdk[d].From(cdk.data() + d * K, K);
                cdk_nnz += sparse_cdk[d].Size();
            }
            for (int w = 0; w < corpus.V; w++)
                SampleWord(w);
        }

        ComputePhi();
        double perplexity = Perplexity();
        cout << "Iteration " << iter
             << " perplexity " << perplexity
             << " nSV " << nSV << " nSVMIters " << nSVMIters
             << " nReject " << num_reject
             << " nnz " << doc_prob_nnz << ' ' << cdk_nnz
             << " time " << svmTime << " " << classTime << " " << ldaTime
             << " num " << num_1 << " " << num_2 << " " << num_3
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

        corpus.ForDoc(d, [&](int w, int z) {
            double l = 0;
            for (int k = 0; k < K; k++)
                l += theta[k] * phi[w * K + k];

            log_likelihood += log(l);
        });
    }
    return exp(-log_likelihood / corpus.T);
}
