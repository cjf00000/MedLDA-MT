//
// Created by Jianfei on 2019/2/27.
//

#ifndef MEDLDA_CORPUS_H
#define MEDLDA_CORPUS_H

#include <string>
#include <vector>
#include <string>
#include <map>

struct Token {
    int w, z;
};

class Corpus {
public:
    Corpus(const std::string &path, Corpus *trainCorpus = nullptr, bool multi_label = false);

    std::vector<std::vector<Token>> w;
    std::vector<int> y;
    std::vector<std::vector<int>> ys;
    std::vector<std::string> vocab;
    std::map<std::string, int> word_to_id;

    void SaveArray(const std::string &data_file);
    void LoadArray(const std::string &data_file);
    void Save(const std::string &data_file);
    bool Load(const std::string &data_file);
    double Accuracy(std::vector<std::vector<double>> &pred);

    bool multi_label;
    int num_docs, V, num_classes;
    size_t T;
};

#endif //MEDLDA_CORPUS_H
