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

Corpus::Corpus(const std::string &data_file, Corpus *trainCorpus, bool multi_label)
    : multi_label(multi_label)
{
    if (trainCorpus) {
        word_to_id = trainCorpus->word_to_id;
        vocab = trainCorpus->vocab;
        V = trainCorpus->V;
        num_classes = trainCorpus->num_classes;
    }
    if (!Load(data_file)) {
        T = 0;
        num_docs = 0;
        if (!trainCorpus) V = 0;

        ifstream f_data(data_file.c_str());
        std::string line;
        while (getline(f_data, line)) {
            std::vector<int> doc;
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
                    if (!trainCorpus) {
                        id = word_to_id[word] = V++;
                        vocab.push_back(word);
                    } else
                        continue;
                }
                id = word_to_id[word];

                while (cnt--) doc.push_back(id);
            }
            T += doc.size();
            num_docs++;
            w.push_back(move(doc));
        }
        d.resize(V);
        for (int i = 0; i < num_docs; i++) {
            for (auto ww: w[i])
                d[ww].push_back(i);
        }
        Save(data_file);
    }

    cout << "Read " << num_docs << " docs, " << T << " tokens.\n"
         << "Vocabulary size is " << V << endl;

    if (!trainCorpus) {
        num_classes = 0;
        for (auto c: y)
            num_classes = max(c+1, num_classes);
    }

    ys.resize(num_classes);
    for (int i = 0; i < num_docs; i++) {
        for (int c = 0; c < num_classes; c++)
            if (c == y[i])
                ys[c].push_back(1);
            else
                ys[c].push_back(-1);
    }
}

void Corpus::AllocZDoc(int K) {
    z.resize(w.size());
    for (int d = 0; d < num_docs; d++) {
        z[d].resize(w[d].size());
        for (auto &k: z[d])
            k = generator() % K;
    }
}

void Corpus::AllocZWord(int K) {
    for (int w = 0; w < V; w++) {
        dz[w].resize(d[w].size());
        for (auto &k: dz[w])
            k = generator() % K;
    }
}

void Corpus::SaveArray(const std::string &data_file, std::vector<std::vector<int>> &arr) {
    ofstream fout(data_file, ios::binary);
    std::vector<int> sizes;
    sizes.push_back(num_docs);
    sizes.push_back(arr.size());
    for (auto &a: arr) sizes.push_back(a.size());
    fout.write((char*)sizes.data(), sizes.size()*sizeof(int));
    fout.write((char*)y.data(), y.size()*sizeof(int));

    for (auto &a: arr)
        fout.write((char*)a.data(), a.size()*sizeof(int));
}

void Corpus::LoadArray(const std::string &data_file, std::vector<std::vector<int>> &arr) {
    ifstream fin(data_file, ios::binary);
    std::vector<int> sizes;
    int a_size;
    fin.read((char*)&num_docs, sizeof(int));
    fin.read((char*)&a_size, sizeof(int));

    sizes.resize(a_size);
    fin.read((char*)sizes.data(), a_size*sizeof(int));
    y.resize(num_docs);
    fin.read((char*)y.data(), num_docs*sizeof(int));
    arr.resize(a_size);
    for (int i = 0; i < a_size; i++)
        arr[i].resize(sizes[i]);
    for (auto &a: arr)
        fin.read((char*)a.data(), a.size()*sizeof(int));
}

bool Corpus::Load(const std::string &data_file) {
    auto bin_d_file = data_file + ".bin.d";
    auto bin_w_file = data_file + ".bin.w";
    auto vocab_file = data_file + ".vocab";
    if (!is_file_exist(bin_d_file.c_str()))
        return false;

    cout << "Found existing data. Loading..." << endl;
    ifstream f_vocab(vocab_file);
    string word;
    V = 0;
    while (f_vocab >> word) {
        vocab.push_back(word);
        word_to_id[word] = V++;
    }

    LoadArray(bin_d_file, d);
    LoadArray(bin_w_file, w);
    T = 0;
    for (auto &doc: w)
        T += doc.size();
    return true;
}

void Corpus::Save(const std::string &data_file) {
    auto bin_d_file = data_file + ".bin.d";
    auto bin_w_file = data_file + ".bin.w";
    auto vocab_file = data_file + ".vocab";

    SaveArray(bin_d_file, d);
    SaveArray(bin_w_file, w);
    ofstream f_vocab(vocab_file);
    for (auto &word: vocab)
        f_vocab << word << "\n";
}

double Corpus::Accuracy(std::vector<std::vector<double>> &pred) {
    int num_correct = 0;
    for (int i = 0; i < num_docs; i++) {
        int max_pred = 0;
        for (int c = 1; c < num_classes; c++)
            if (pred[c][i] > pred[max_pred][i])
                max_pred = c;
         if (max_pred == y[i])
             num_correct++;
    }
    return (double)num_correct / num_docs;
}