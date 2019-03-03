//
// Created by Jianfei on 2019/2/27.
//

#ifndef MEDLDA_CORPUS_H
#define MEDLDA_CORPUS_H

#include <string>
#include <vector>
#include <string>
#include <map>
#include <random>

struct Token {
    int w, z;
};

struct DToken {
    int d, z;
};

class Corpus {
public:
    Corpus(const std::string &path, Corpus *trainCorpus = nullptr, bool multi_label = false);

    std::vector<std::vector<int>> w, d, z, dz;
    std::vector<std::vector<int>> ys;
    std::vector<std::string> vocab;
    std::map<std::string, int> word_to_id;

    void AllocZDoc(int K);
    void AllocZWord(int K);

    template <class Func>
    void ForDoc(int d, Func f) {
        for (size_t n = 0; n < w[d].size(); n++)
            f(w[d][n], z[d][n]);
    }

    template <class Func>
    void ForWord(int w, Func f) {
        for (size_t n = 0; n < d[w].size(); n++)
            f(d[w][n], dz[w][n]);
    }

    void SaveArray(const std::string &data_file, std::vector<std::vector<int>> &a, int size);
    void LoadArray(const std::string &data_file, std::vector<std::vector<int>> &a, int &size);
    void Save(const std::string &data_file); // num_docs, V, num_classes, vocab, w, d, ys
    bool Load(const std::string &data_file);
    double Accuracy(std::vector<std::vector<double>> &pred);
    std::pair<double, double> F1(std::vector<std::vector<double>> &pred);

    bool multi_label;
    int num_docs, V, num_classes;
    size_t T;
    std::mt19937 generator;
};

#endif //MEDLDA_CORPUS_H
