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
        vector<vector<int>> labels;

        ifstream f_data(data_file.c_str());
        std::string line;
        while (getline(f_data, line)) {
            std::vector<int> doc;
            std::vector<int> label_doc;

            // Read labels
            size_t first_colon;
            for (first_colon = 0; first_colon < line.size(); first_colon++)
                if (line[first_colon] == ':')
                    break;
            if (first_colon != line.size()) {
                while (first_colon > 0 && line[first_colon] != ' ')
                    first_colon--;
            }
            istringstream label_in(string(line.begin(), line.begin()+first_colon));
            int label;
            while (label_in >> label) {
                if (!trainCorpus || label < num_classes)
                    label_doc.push_back(label);
            }
            labels.push_back(move(label_doc));

            for (size_t idx = 0; idx < first_colon; idx++) line[idx] = ' ';

            for (auto &c: line)
                if (c == ':')
                    c = ' ';

            int cnt;
            string word;
            istringstream sin(line);
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
        if (!trainCorpus) {
            num_classes = 0;
            for (auto &ld: labels)
                for (auto l: ld)
                    num_classes = max(l+1, num_classes);
        }
        ys.resize(num_classes);
        for (auto &y: ys) {
            y.resize(num_docs);
            for (auto &l: y) l = -1;
        }
        for (int i = 0; i < num_docs; i++) {
            for (auto l: labels[i])
                ys[l][i] = 1;
        }

        Save(data_file);
    }

    cout << "Read " << num_docs << " docs, " << T << " tokens " << num_classes << " classes.\n"
         << "Vocabulary size is " << V << endl;
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
    dz.resize(d.size());
    for (int w = 0; w < V; w++) {
        dz[w].resize(d[w].size());
        for (auto &k: dz[w])
            k = generator() % K;
    }
}

void Corpus::SaveArray(const std::string &data_file, std::vector<std::vector<int>> &arr, int size) {
    ofstream fout(data_file, ios::binary);
    std::vector<int> sizes;
    sizes.push_back(size);
    for (auto &a: arr) sizes.push_back(a.size());
    fout.write((char*)sizes.data(), sizes.size()*sizeof(int));

    for (auto &a: arr)
        fout.write((char*)a.data(), a.size()*sizeof(int));
}

void Corpus::LoadArray(const std::string &data_file, std::vector<std::vector<int>> &arr, int &size) {
    ifstream fin(data_file, ios::binary);
    std::vector<int> sizes;
    fin.read((char*)&size, sizeof(int));

    sizes.resize(size);
    fin.read((char*)sizes.data(), size*sizeof(int));
    arr.resize(size);
    for (int i = 0; i < size; i++)
        arr[i].resize(sizes[i]);
    for (auto &a: arr)
        fin.read((char*)a.data(), a.size()*sizeof(int));
}

bool Corpus::Load(const std::string &data_file) {
    auto bin_d_file = data_file + ".bin.d";
    auto bin_w_file = data_file + ".bin.w";
    auto bin_y_file = data_file + ".bin.y";
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

    LoadArray(bin_d_file, d, V);
    LoadArray(bin_w_file, w, num_docs);
    LoadArray(bin_y_file, ys, num_classes);
    T = 0;
    for (auto &doc: w)
        T += doc.size();
    return true;
}

void Corpus::Save(const std::string &data_file) {
    auto bin_d_file = data_file + ".bin.d";
    auto bin_w_file = data_file + ".bin.w";
    auto bin_y_file = data_file + ".bin.y";
    auto vocab_file = data_file + ".vocab";

    SaveArray(bin_d_file, d, V);
    SaveArray(bin_w_file, w, num_docs);
    SaveArray(bin_y_file, ys, num_classes);
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
         if (ys[max_pred][i] > 0)
             num_correct++;
    }
    return (double)num_correct / num_docs;
}


std::pair<double, double> Corpus::F1(std::vector<std::vector<double>> &pred) {
    float macro_f1 = 0;
    int total_tp = 0, total_fp = 0, total_fn = 0;

    auto GetF1 = [&](int tp, int fp, int fn) {
        float precision = (float)tp / (tp + fp);
        if (tp + fp == 0) precision = 0;
        float recall = (float)tp / (tp + fn);
        if (tp + fn == 0) recall = 0;
        float f1 = 2 * (precision * recall) / (precision + recall);
        if (precision == 0 || recall == 0) f1 = 0;
        return f1;
    };

    for (int c = 0; c < num_classes; c++) {
        int tp = 0, fp = 0, fn = 0;
        for (int i = 0; i < num_docs; i++) {
            int p = -1;
            if (pred[c][i] > 0) p = 1;
            if (p>0 && ys[c][i]>0) tp++;
            if (p<0 && ys[c][i]>0) fn++;
            if (p>0 && ys[c][i]<0) fp++;
        }

        macro_f1 += GetF1(tp, fp, fn);
        total_tp += tp;
        total_fp += fp;
        total_fn += fn;
    }
    float micro_f1 = GetF1(total_tp, total_fp, total_fn);
    macro_f1 /= num_classes;
    return make_pair(micro_f1, macro_f1);
}