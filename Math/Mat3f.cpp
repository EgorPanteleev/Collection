//
// Created by igor on 16.01.2024.
//

#include "Mat3f.h"
#include "Utils.h"
Vector3f& Mat3f::operator[]( int index ) {
    return columns[index];
}

const Vector3f& Mat3f::operator[]( int index ) const {
    return columns[index];
}

Mat3f::Mat3f(): columns() { }

Mat3f::Mat3f( const Vector3f& vec1, const Vector3f& vec2, const Vector3f& vec3 ) {
    columns[0] = vec1;
    columns[1] = vec2;
    columns[2] = vec3;
}

float Mat3f::getDet() const {
    return ( columns[0][0] * columns[1][1] * columns[2][2] +
             columns[1][0] * columns[2][1] * columns[0][2] +
             columns[0][1] * columns[1][2] * columns[2][0] -
             columns[0][2] * columns[1][1] * columns[2][0] -
             columns[0][1] * columns[1][0] * columns[2][2] -
             columns[1][2] * columns[2][1] * columns[0][0]);
}

Mat3f Mat3f::transpose() const {
    Mat3f res;
    int i = 0;
    for ( auto c : columns ) {
        res[0][i] = c[0];
        res[1][i] = c[1];
        res[2][i] = c[2];
        i++;
    }
    return res;
}

Mat3f Mat3f::inverse() const {
    Mat3f res;
    res = transpose() / getDet();
    return res;
}
