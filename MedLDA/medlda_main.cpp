//
// Created by Jianfei on 2019/2/27.
//
#include "gflags/gflags.h"
#include "corpus.h"
#include "medlda.h"
#include <iostream>
using namespace std;

DEFINE_string(data, "../data/20news", "Prefix of the dataset");
DEFINE_int32(K, 100, "Number of topics");
DEFINE_double(alpha_sum, 50, "Sum of alpha");
DEFINE_double(beta, 0.01, "Sum of beta");
DEFINE_double(C, 1, "SVM cost");
DEFINE_double(ell, 1, "SVM margin");
DEFINE_double(tol, 0.1, "Tolerance of SVM solver");
DEFINE_bool(multi_label, false, "Multi-label or multi-class");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    cout << "---------------------------------------------------------------------" << endl;
    std::vector<gflags::CommandLineFlagInfo> all_flags;
    gflags::GetAllFlags(&all_flags);
    for (const auto& flag : all_flags) {
        if (flag.filename.find("src/") != 0) // HACK: filter out built-in flags
            cout << flag.name << ": " << flag.current_value << endl;
    }
    cout << "---------------------------------------------------------------------" << endl;

    Corpus corpus(FLAGS_data + ".train", nullptr, FLAGS_multi_label);
    Corpus testCorpus(FLAGS_data + ".test", &corpus, FLAGS_multi_label);
    MedLDA model(corpus, testCorpus, FLAGS_K, FLAGS_alpha_sum / FLAGS_K, FLAGS_beta, FLAGS_C, FLAGS_ell, FLAGS_tol);
    model.Train();
}
