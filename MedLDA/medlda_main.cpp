//
// Created by Jianfei on 2019/2/27.
//
#include "gflags/gflags.h"
#include "corpus.h"
#include "medlda.h"
using namespace std;

DEFINE_string(data, "../data/20news", "Prefix of the dataset");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    Corpus corpus(FLAGS_data + ".train");
    Corpus testCorpus(FLAGS_data + ".test", &corpus);
    MedLDA model(corpus, testCorpus, 100, 0.5, 0.01, 1.0, 1.0, 0.1);
    model.Train();
}
