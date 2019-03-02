#include "gflags/gflags.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include "svm.h"
#include "corpus.h"
using namespace std;

DEFINE_string(data, "../data/20news", "Prefix of the dataset");
DEFINE_double(C, 1.0, "C");
DEFINE_double(ell, 1.0, "Margin");
DEFINE_double(eps, 0.1, "Tolerance");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    Corpus train(FLAGS_data + ".train");
    Corpus test(FLAGS_data + ".test", &train);
    cout << "Read " << train.num_docs << " instances, "
         << train.V << " features "
         << train.num_classes << " classes" << endl;
    cout << "Read " << test.num_docs << " instances, "
         << test.V << " features "
         << test.num_classes << " classes" << endl;

    auto ConvertX = [&](Corpus &corpus, vector<Feature> &X) {
        Feature feature;
        for (auto &doc: corpus.w) {
            feature.clear();
            sort(doc.begin(), doc.end());
            int N = (int)doc.size();
            int j = 0;
            for (int i = 0; i < N; i = j) {
                for (j = i; j < N && doc[i] == doc[j]; j++);
                feature.push_back(Entry{doc[i], (float)(j - i)});
            }
            X.push_back(feature);
        }
    };
    vector<Feature> train_X, test_X;
    ConvertX(train, train_X);
    ConvertX(test, test_X);

    vector<SVM> models;
    vector<vector<double>> train_p, test_p;
    for (int c = 0; c < train.num_classes; c++) {
        models.push_back(SVM(train.num_docs, train.V, FLAGS_C, FLAGS_ell, FLAGS_eps));
        models.back().Solve(train_X, train.ys[c]);
        cout << models.back().nSV() << " SVs." << endl;
        train_p.push_back(models.back().Predict(train_X));
        test_p.push_back(models.back().Predict(test_X));
    }

    cout << "Training accuracy " << train.Accuracy(train_p)
         << " Testing accuracy " << test.Accuracy(test_p) << endl;
}
