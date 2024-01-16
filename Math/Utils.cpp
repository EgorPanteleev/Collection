#include "Utils.h"

double dot( Vector3f p1, Vector3f p2 ) {
    return ( p1.getX() * p2.getX() + p1.getY() * p2.getY() + p1.getZ() * p2.getZ());
}

double dot( Vector4f p1, Vector4f p2 ) {
    return ( p1.getX() * p2.getX() + p1.getY() * p2.getY() + p1.getZ() * p2.getZ() + p1.getW() * p2.getW());
}

Mat4f operator*( float a, const Mat4f& m ) {
    return Mat4f(
            m[0] * a,
            m[1] * a,
            m[2] * a,
            m[3] * a
    );
}

Mat4f operator*( const Mat4f& m, float a ) {
    return Mat4f(
            m[0] * a,
            m[1] * a,
            m[2] * a,
            m[3] * a
    );
}

Vector4f operator*( const Mat4f& m, const Vector4f& v ) {
    return Vector4f(
            m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0] * v[3],
            m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1] * v[3],
            m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2] * v[3],
            m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3] * v[3]
    );
}

Vector4f operator*( const Vector4f& v, const Mat4f& m ) {
    return Vector4f(
            v[0] * m[0][0] + v[1] * m[0][1] + v[2] * m[0][2] + v[3] * m[0][3],
            v[0] * m[1][0] + v[1] * m[1][1] + v[2] * m[1][2] + v[3] * m[1][3],
            v[0] * m[2][0] + v[1] * m[2][1] + v[2] * m[2][2] + v[3] * m[2][3],
            v[0] * m[3][0] + v[1] * m[3][1] + v[2] * m[3][2] + v[3] * m[3][3]
    );
}

Mat4f operator*( const Mat4f& m1, const Mat4f& m2 ) {
    Vector4f vec1 = m1 * m2[0];
    Vector4f vec2 = m1 * m2[1];
    Vector4f vec3 = m1 * m2[2];
    Vector4f vec4 = m1 * m2[3];
    return Mat4f( vec1, vec2, vec3, vec4 );
}