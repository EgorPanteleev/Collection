#include <iostream>
#include "Mat.h"
#include <cmath>
int testDet() {
    Mat3f mat3 = {
            {1,2,3},
          {2,3,4},
          {4,5,6}};
    if ( mat3.getDet() != 0 ) return 1;
    mat3 =  {
            {1.2,2.5,3.6},
            {2.2,3.2,4.1},
            {4.0,5.9,6.2}};
    auto asd = round(mat3.getDet()*1000)/1000;
    if ( round(mat3.getDet()*1000) != 2328 ) return 1;
    return 0;
}

int testTranspose() {
    Mat3f mat3 = {
            {1,2,3},
            {2,3,4},
            {4,5,6}};
    Mat3f mat3Transposed = {
            {1,2,4},
            {2,3,5},
            {3,4,6}};
    if ( mat3.transpose() != mat3Transposed ) return 1;
    mat3 =  {
            {1.2,2.5,3.6},
            {2.2,3.2,4.1},
            {4.0,5.9,6.2}};
    mat3Transposed = {
            {1.2,2.2,4.0},
            {2.5,3.2,5.9},
            {3.6,4.1,6.2}};
    if ( mat3.transpose() != mat3Transposed ) return 1;
    return 0;
}

int testInverse() {
    Mat3f mat3 = {
            {1,2,3},
            {2,3,4},
            {4,5,7}};
    Mat3f mat3Inversed = {
            {-1,-1,1},
            {-2,5,-2},
            {2,-3,1}};
    if ( mat3.inverse() != mat3Inversed ) return 1;
    return 0;
}

int testRotationMatrix() {

}

int main() {
    int res = 0;
    res += testDet();
    res += testTranspose();
    res += testInverse();
    return res;
}
