//
// Created by Jianfei on 2019/2/25.
//

#include "svm.h"
#include <algorithm>
#include <random>
#include <iostream>
using namespace std;

DEFINE_int32(max_svm_iters, 10000, "Max number of SVM iterations per coordinate descent iteration.");

SVM::SVM(int num_data, int num_features, double C, double ell, double eps)
        : num_data(num_data), num_features(num_features), C(C), ell(ell), eps(eps),
        diag(num_data), w(num_features), alpha(num_data), perm(num_data)
{
    iota(perm.begin(), perm.end(), 0);
}

void SVM::Solve(std::vector<Feature> &X, std::vector<int> &y) {
    diag.clear(); // diag[i] = <x[i], x[i]>
    for (auto &x: X) {
        double sum = 0;
        for (auto &e: x)
            sum += e.v * e.v;
        diag.push_back(sum);
    }
    fill(w.begin(), w.end(), 0);
    for (int i = 0; i < num_data; i++)
        for (auto &e: X[i])
            w[e.k] += y[i] * alpha[i] * e.v;

    double old_obj = 1e9;
    int iter = 0;
    while (1) {
        iter += 1;
        shuffle(perm.begin(), perm.end(), generator);

        for (int i: perm) {  // grad = y[i]<X[i], w> - 1
            double grad = 0;
            for (auto &e: X[i])
                grad += w[e.k] * e.v;
            grad = grad * y[i] - ell;

            double pg = grad;
            if (alpha[i] < eps) pg = min(pg, 0.);
            if (alpha[i] > C - eps) pg = max(pg, 0.);

            if (fabs(pg) >= eps) {
                double old_alpha = alpha[i];
                double new_alpha = max(0., min(C, old_alpha - grad / diag[i]));
                double d_alpha = new_alpha - old_alpha;
                alpha[i] = new_alpha;
                for (auto &e: X[i])
                    w[e.k] += d_alpha * y[i] * e.v;
            }
        }

        double obj = 0;
        for (auto wi: w) obj += wi * wi;
        obj *= 0.5;
        for (auto ai: alpha) obj -= ell * ai;

//        cout << "Iter " << iter << " Objective function " << obj << endl;

        if (fabs(old_obj - obj) < eps || iter >= FLAGS_max_svm_iters)
            break;
        old_obj = obj;
    }
    num_iters = iter;
}

int SVM::nSV() {
    int cnt = 0;
    for (auto ai: alpha)
        if (fabs(ai) > 1e-7)
            cnt++;

    return cnt;
}

std::vector<double> SVM::Predict(std::vector<Feature> &X) {
    std::vector<double> result(X.size());
    int num_data = X.size();
    for (int i = 0; i < num_data; i++) {
        double p = 0;
        for (auto &e: X[i]) p += w[e.k] * e.v;
        result[i] = p;
    }
    return move(result);
}
