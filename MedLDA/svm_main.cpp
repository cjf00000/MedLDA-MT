#include "gflags/gflags.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include "svm.h"
using namespace std;

DEFINE_string(train_data, "../data/nips", "Prefix of the dataset");
DEFINE_string(test_data, "../data/nips", "Prefix of the dataset");
DEFINE_double(C, 1.0, "C");
DEFINE_double(ell, 1.0, "Margin");
DEFINE_double(eps, 0.1, "Tolerance");

struct SVMData {
    SVMData(string path) {
        ifstream fin(path);
        string line;
        num_features = 0;
        while (getline(fin, line)) {
            for (char &ch: line) if (ch == ':') ch = ' ';
            int y, k; float v;
            istringstream sin(line);
            sin >> y;
            Feature feature;
            while (sin >> k >> v) {
                num_features = max(num_features, k);
                feature.push_back(Entry{k-1, v});
            }
            X.push_back(std::move(feature));
            Y.push_back(y);
        }
    }

    std::vector<Feature> X;
    std::vector<int> Y;
    int num_features;
};

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    SVMData train(FLAGS_train_data);
    SVMData test(FLAGS_test_data);
    cout << "Read " << train.X.size() << " instances, " << train.num_features << " features." << endl;
    cout << "Read " << test.X.size() << " instances, " << test.num_features << " features." << endl;
    SVM model(train.X.size(), train.num_features, FLAGS_C, FLAGS_ell, FLAGS_eps);
    model.SetData(train.X, train.Y);
    model.Solve();
    cout << model.nSV() << " SVs." << endl;
    cout << "Training accuracy " << model.Score(train.X, train.Y)
         << " Testing accuracy " << model.Score(test.X, test.Y) << endl;
}
