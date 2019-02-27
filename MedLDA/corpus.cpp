//
// Created by Jianfei on 2019/2/27.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include "corpus.h"
using namespace std;

bool is_file_exist(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

Corpus::Corpus(const std::string &data_file) {
    if (!Load(data_file)) {
        train_T = 0;
        test_T = 0;
        num_train = 0;
        num_test = 0;
        V = 0;

        auto ReadCorpus = [&](string path, bool is_train, int &num_docs, size_t &T) {
            ifstream f_data(path.c_str());
            std::string line;
            while (getline(f_data, line)) {
                std::vector<Token> doc;
                for (auto &c: line)
                    if (c == ':')
                        c = ' ';

                int label, cnt;
                string word;
                istringstream sin(line);
                sin >> label;
                y.push_back(label);
                while (sin >> word >> cnt) {
                    int id;
                    if (word_to_id.find(word) == word_to_id.end()) {
                        if (is_train) {
                            id = word_to_id[word] = V++;
                            vocab.push_back(word);
                        } else
                            continue;
                    } else
                        id = word_to_id[word];

                    while (cnt--) doc.push_back(Token{id, 0});
                }
                T += doc.size();
                num_docs++;
                w.push_back(move(doc));
            }
        };
        ReadCorpus(data_file + ".train", true, num_train, train_T);
        ReadCorpus(data_file + ".test", false, num_test, test_T);
        Save(data_file);
    }

    cout << "Read " << num_train << " training docs, " << train_T << " tokens.\n"
         << "Read " << num_test << " testing docs, " << test_T << " tokens.\n"
         << "Vocabulary size is " << V << endl;
}

void Corpus::SaveArray(const std::string &data_file) {
    ofstream fout(data_file, ios::binary);
    std::vector<int> sizes;
    sizes.push_back(num_train);
    sizes.push_back(num_test);
    for (auto &a: w) sizes.push_back(a.size());
    fout.write((char*)sizes.data(), sizes.size()*sizeof(int));
    fout.write((char*)y.data(), y.size()*sizeof(int));

    for (auto &a: w)
        fout.write((char*)a.data(), a.size()*sizeof(Token));
}

void Corpus::LoadArray(const std::string &data_file) {
    ifstream fin(data_file, ios::binary);
    std::vector<int> sizes;
    fin.read((char*)&num_train, sizeof(int));
    fin.read((char*)&num_test, sizeof(int));
    int N = num_train + num_test;
    sizes.resize(N);
    fin.read((char*)sizes.data(), N*sizeof(int));
    y.resize(N);
    fin.read((char*)y.data(), N*sizeof(int));
    w.resize(N);
    for (int i = 0; i < N; i++)
        w[i].resize(sizes[i]);
    for (auto &a: w)
        fin.read((char*)a.data(), a.size()*sizeof(Token));
}

bool Corpus::Load(const std::string &data_file) {
    auto bin_file = data_file + ".bin";
    auto vocab_file = data_file + ".vocab";
    if (!is_file_exist(bin_file.c_str()))
        return false;

    cout << "Found existing data. Loading..." << endl;
    ifstream f_vocab(vocab_file);
    string word;
    V = 0;
    while (f_vocab >> word) {
        vocab.push_back(word);
        word_to_id[word] = V++;
    }

    LoadArray(bin_file);

    train_T = test_T = 0;
    for (int i = 0; i < num_train; i++)
        train_T += w[i].size();
    for (int i = num_train; i < num_train + num_test; i++)
        test_T += w[i].size();
    return true;
}

void Corpus::Save(const std::string &data_file) {
    auto bin_file = data_file + ".bin";
    auto vocab_file = data_file + ".vocab";

    SaveArray(bin_file);
    ofstream f_vocab(vocab_file);
    for (auto &word: vocab)
        f_vocab << word << "\n";
}