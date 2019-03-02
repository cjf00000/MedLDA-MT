//
// Created by Jianfei on 2019/2/25.
//

#ifndef MEDLDA_MT_SVM_H
#define MEDLDA_MT_SVM_H

#include <vector>
#include <random>
#include "utils.h"
#include "gflags/gflags.h"

DECLARE_int32(max_svm_iters);

typedef std::vector<Entry> Feature;

class SVM {
public:
    SVM() {}
    SVM(int num_data, int num_features, double C, double ell, double eps = 1e-7);

    // Input: X, y, initial alpha, w
    // Output: alpha, w
    void Solve(std::vector<Feature> &X, std::vector<int> &y);

    int nSV();

    std::vector<double> Predict(std::vector<Feature> &X);

    std::vector<double> alpha;
    std::vector<double> w;
    int num_iters;

private:
    std::vector<double> diag;
    std::vector<int> perm;
    int num_features, num_data;
    double C, ell;
    double eps;
    std::mt19937 generator;
};

#endif //MEDLDA_MT_SVM_H
