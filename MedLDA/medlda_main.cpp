//
// Created by Jianfei on 2019/2/27.
//
#include "gflags/gflags.h"
#include "corpus.h"
using namespace std;

DEFINE_string(data, "../data/20news", "Prefix of the dataset");

int main(int argc, char **argv) {
    Corpus corpus(FLAGS_data);
}
