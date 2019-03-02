//
// Created by Jianfei on 2019/2/27.
//
#include "gflags/gflags.h"
#include "corpus.h"
#include "medlda.h"
using namespace std;

DEFINE_string(data, "../data/20news", "Prefix of the dataset");
DEFINE_int32(K, 100, "Number of topics");
DEFINE_double(alpha_sum, 50, "Sum of alpha");
DEFINE_double(beta, 0.01, "Sum of beta");
DEFINE_double(C, 1, "SVM cost");
DEFINE_double(ell, 1, "SVM margin");
DEFINE_double(tol, 0.1, "Tolerance of SVM solver");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    Corpus corpus(FLAGS_data + ".train");
    Corpus testCorpus(FLAGS_data + ".test", &corpus);
    MedLDA model(corpus, testCorpus, FLAGS_K, FLAGS_alpha_sum / FLAGS_K, FLAGS_beta, FLAGS_C, FLAGS_ell, FLAGS_tol);
    model.Train();
}
