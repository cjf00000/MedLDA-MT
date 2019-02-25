#include "gflags/gflags.h"
#include <iostream>
using namespace std;

DEFINE_string(alg, "ds", "cgs or ds or em or goem or scvb0");
DEFINE_string(prefix, "../data/nips", "Prefix of the dataset");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    cout << FLAGS_alg << endl;
}
