//
// Created by igor on 16.01.2024.
//

#include "Mat4f.h"

Vector4f& Mat4f::operator[]( int index ) {
    return columns[index];
}

const Vector4f& Mat4f::operator[]( int index ) const {
    return columns[index];
}

Mat4f::Mat4f(): columns() { }

Mat4f::Mat4f( Vector4f vec1, Vector4f vec2, Vector4f vec3, Vector4f vec4 ) {
    columns[0] = vec1;
    columns[1] = vec2;
    columns[2] = vec3;
    columns[3] = vec4;
}