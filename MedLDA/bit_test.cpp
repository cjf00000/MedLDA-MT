//
// Created by Jianfei on 2019/3/2.
//
#include "bit.h"
#include <iostream>
using namespace std;

int main() {
    BIT bit;
    bit.Resize(5);      // 1, 3, 6, 10, 15
    bit.Update(0, 1);
    bit.Update(1, 2);
    bit.Update(2, 3);
    bit.Update(3, 4);
    bit.Update(4, 5);
    for (int i = 0; i < 9; i++)
        cout << bit.a[i] << ' ';
    cout << endl;
    cout << bit.N << ' ' << bit.M << endl;
    cout << bit.GetIndex(0) << endl;
    cout << bit.GetIndex(1) << endl;
    cout << bit.GetIndex(2) << endl;
    cout << bit.GetIndex(5) << endl;
    cout << bit.GetIndex(8) << endl;
    cout << bit.GetIndex(13) << endl;
    cout << bit.GetIndex(16) << endl;
}