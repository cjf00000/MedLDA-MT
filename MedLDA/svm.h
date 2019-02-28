//
// Created by Jianfei on 2019/2/25.
//

#ifndef MEDLDA_MT_SVM_H
#define MEDLDA_MT_SVM_H

#include <vector>
#include <random>

struct Entry {
    int k;
    float v;
};

typedef std::vector<Entry> Feature;

class SVM {
public:
    SVM(int num_data, int num_features, double C, double ell, double eps = 1e-7);

    void SetData(std::vector<Feature> &X, std::vector<int> &y, bool move = false);

    // Input: X, y, initial alpha, w
    // Output: alpha, w
    void Solve();

    int nSV();

    double Score(std::vector<Feature> &X, std::vector<int> &y);

private:
    std::vector<Feature> X;
    std::vector<int> y;
    std::vector<double> diag;
    std::vector<double> w;
    std::vector<double> alpha;
    std::vector<int> perm;
    int num_features, num_data;
    double C, ell;
    double eps;
    std::mt19937 generator;
};

#endif //MEDLDA_MT_SVM_H
