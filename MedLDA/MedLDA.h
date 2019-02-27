//
// Created by Jianfei on 2019/2/27.
//

#ifndef MEDLDA_MEDLDA_H
#define MEDLDA_MEDLDA_H

#include <vector>

class MedLDA {
public:
    MedLDA(Corpus &corpus, float alpha, float beta, float ell, float C);

private:
    Corpus& corpus;
    float alpha, beta, ell, C;
    std::vector<int> cdk, cwk, ck;
    std::vector<float> phi, inv_ck;
};


#endif //MEDLDA_MEDLDA_H
