//
// Created by igor on 16.01.2024.
//

#include "Mat3f.h"
#include "Utils.h"
#include "Mat2f.h"
Vector3f& Mat3f::operator[]( int index ) {
    return columns[index];
}

const Vector3f& Mat3f::operator[]( int index ) const {
    return columns[index];
}

bool Mat3f::operator==( Mat3f& mat ) const {
    for ( int i = 0; i < 3; ++i ) {
        for ( int j = 0; j < 3; ++j ) {
            if ( columns[i][j] != mat[i][j] ) return false;
        }
    }
    return true;
}

bool Mat3f::operator!=( Mat3f& mat ) const {
    return ( ! (*this == mat) );
}
Mat3f::Mat3f(): columns() { }

Mat3f::Mat3f( const Vector3f& vec1, const Vector3f& vec2, const Vector3f& vec3 ) {
    columns[0] = vec1;
    columns[1] = vec2;
    columns[2] = vec3;
}

float Mat3f::getAlgExtension( int col, int row ) const {
    Mat2f mat2f;
    float sign = ( ( col + row ) % 2 == 0 ) ? 1 : -1;
    for ( int i = 0, si = 0; i < 3; ++i ) {
        if ( i == col ) continue;
        for ( int j = 0, sj = 0; j < 3; ++j ) {
            if ( j == row ) continue;
            mat2f[si][sj] = columns[i][j];
            ++sj;
        }
        ++si;
    }
    return sign * mat2f.getDet();
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
    float det = getDet();
    if ( det == 0 ) return {};
    res = getUnion().transpose() / det;
    return res;
}

Mat3f Mat3f::getUnion() const {
    return {{getAlgExtension(0, 0), getAlgExtension(0, 1), getAlgExtension(0, 2) },
            {getAlgExtension(1, 0), getAlgExtension(1, 1), getAlgExtension(1, 2) },
            {getAlgExtension(2, 0), getAlgExtension(2, 1), getAlgExtension(2, 2) }};
}
