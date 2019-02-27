//
// Created by Jianfei on 2019/2/27.
//
#include "gflags/gflags.h"
#include "corpus.h"
#include "medlda.h"
using namespace std;

DEFINE_string(data, "../data/20news", "Prefix of the dataset");

int main(int argc, char **argv) {
    Corpus corpus(FLAGS_data);
    MedLDA model(corpus, 100, 0.5, 0.01, 1.0, 1.0);
    model.Train();
}
