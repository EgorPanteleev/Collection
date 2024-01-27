//
// Created by igor on 16.01.2024.
//

#include "Mat4f.h"
#include "Mat3f.h"
#include "Utils.h"
Vector4f& Mat4f::operator[]( int index ) {
    return columns[index];
}

const Vector4f& Mat4f::operator[]( int index ) const {
    return columns[index];
}

Mat4f::Mat4f(): columns() { }

Mat4f::Mat4f( const Vector4f& vec1, const Vector4f& vec2, const Vector4f& vec3, const Vector4f& vec4 ) {
    columns[0] = vec1;
    columns[1] = vec2;
    columns[2] = vec3;
    columns[3] = vec4;
}

float Mat4f::getDet() const {
    Mat3f first = Mat3f(Vector3f(columns[0][1],columns[0][2] ,columns[0][3]),
                      Vector3f(columns[1][1],columns[1][2] ,columns[1][3]),
                      Vector3f(columns[2][1],columns[2][2] ,columns[2][3]));
    Mat3f second = Mat3f(Vector3f(columns[0][0],columns[0][2] ,columns[0][3]),
                        Vector3f(columns[1][0],columns[1][2] ,columns[1][3]),
                        Vector3f(columns[2][0],columns[2][2] ,columns[2][3]));
    Mat3f third = Mat3f(Vector3f(columns[0][0],columns[0][1] ,columns[0][3]),
                        Vector3f(columns[1][0],columns[1][1] ,columns[1][3]),
                        Vector3f(columns[2][0],columns[2][1] ,columns[2][3]));
    Mat3f fourth = Mat3f(Vector3f(columns[0][0],columns[0][1] ,columns[0][2]),
                        Vector3f(columns[1][0],columns[1][1] ,columns[1][2]),
                        Vector3f(columns[2][0],columns[2][1] ,columns[2][2]));
    return (-columns[3][0] * first.getDet() + columns[3][1] * second.getDet() -
             columns[3][2] * third.getDet() + columns[3][3] * fourth.getDet());
}

Mat4f Mat4f::transpose() const {
    Mat4f res;
    int i = 0;
    for ( auto c : columns ) {
        res[0][i] = c[0];
        res[1][i] = c[1];
        res[2][i] = c[2];
        res[3][i] = c[3];
        i++;
    }
    return res;
}

Mat4f Mat4f::inverse() const {
    Mat4f res;
    Mat4f tran = transpose();
    res = tran / getDet();
    return res;
}
