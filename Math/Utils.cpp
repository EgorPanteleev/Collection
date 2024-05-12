#include "Utils.h"
#include <cmath>
float dot( const Vector3f& p1, const Vector3f& p2 ) {
    return ( p1.x * p2.x + p1.y * p2.y + p1.z * p2.z );
}

float dot( const Vector4f& p1, const Vector4f& p2 ) {
    return ( p1.getX() * p2.getX() + p1.getY() * p2.getY() + p1.getZ() * p2.getZ() + p1.getW() * p2.getW());
}

float getDistance( const Vector3f& p1, const Vector3f& p2 ) {
    return ( float ) sqrt(pow((p2.getX() - p1.getX()), 2)
                           + pow((p2.getY() - p1.getY()), 2)
                           + pow((p2.getZ() - p1.getZ()), 2) );
}


Mat4f operator*( float a, const Mat4f& m ) {
    return Mat4f{
            m[0] * a,
            m[1] * a,
            m[2] * a,
            m[3] * a
    };
}

Mat4f operator*( const Mat4f& m, float a ) {
    return Mat4f{
            m[0] * a,
            m[1] * a,
            m[2] * a,
            m[3] * a
    };
}

Mat4f operator/( const Mat4f& m, float a ) {
    Vector4f asd2 = m[2] / a;
    return Mat4f{
            m[0] / a,
            m[1] / a,
            m[2] / a,
            m[3] / a
    };
}

Mat3f operator/( const Mat3f& m, float a ) {
    return Mat3f{
            m[0] / a,
            m[1] / a,
            m[2] / a
    };
}

Mat2f operator/( const Mat2f& m, float a ) {
    return Mat2f{
            m[0] / a,
            m[1] / a
    };
}

Vector4f operator*( const Mat4f& m, const Vector4f& v ) {
    return Vector4f{
            m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0] * v[3],
            m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1] * v[3],
            m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2] * v[3],
            m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3] * v[3]
    };
}

Vector4f operator*( const Vector4f& v, const Mat4f& m ) {
    return Vector4f{
            v[0] * m[0][0] + v[1] * m[0][1] + v[2] * m[0][2] + v[3] * m[0][3],
            v[0] * m[1][0] + v[1] * m[1][1] + v[2] * m[1][2] + v[3] * m[1][3],
            v[0] * m[2][0] + v[1] * m[2][1] + v[2] * m[2][2] + v[3] * m[2][3],
            v[0] * m[3][0] + v[1] * m[3][1] + v[2] * m[3][2] + v[3] * m[3][3]
    };
}

Mat4f operator*( const Mat4f& m1, const Mat4f& m2 ) {
    Vector4f vec1 = m1 * m2[0];
    Vector4f vec2 = m1 * m2[1];
    Vector4f vec3 = m1 * m2[2];
    Vector4f vec4 = m1 * m2[3];
    return Mat4f{ vec1, vec2, vec3, vec4 };
}

Vector3f operator*( const Mat3f& m, const Vector3f& v ) {
    return Vector3f{
            m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
            m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
            m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2]
    };
}

Mat3f operator*( const Mat3f& m1, const Mat3f& m2 ) {
    Vector3f vec1 = m1 * m2[0];
    Vector3f vec2 = m1 * m2[1];
    Vector3f vec3 = m1 * m2[2];
    return Mat3f{ vec1, vec2, vec3 };
}

Mat3f operator*( const Mat3f& m1, float a ) {
    Vector3f vec1 = m1[0] * a;
    Vector3f vec2 = m1[1] * a;
    Vector3f vec3 = m1[2] * a;
    return Mat3f{ vec1, vec2, vec3 };
}

Mat3f operator*( float a, const Mat3f& m1 ) {
    Vector3f vec1 = m1[0] * a;
    Vector3f vec2 = m1[1] * a;
    Vector3f vec3 = m1[2] * a;
    return Mat3f{ vec1, vec2, vec3 };
}

Mat3f operator+( const Mat3f& m1, const Mat3f& m2 ) {
    Vector3f vec1 = m1[0] + m2[0];
    Vector3f vec2 = m1[1] + m2[1];
    Vector3f vec3 = m1[2] + m2[2];
    return Mat3f{ vec1, vec2, vec3 };
}

Vector3f min( const Vector3f& v1, const Vector3f& v2 ) {
    return { std::min( v1.x, v2.x), std::min( v1.y, v2.y), std::min( v1.z, v2.z) };
}

Vector3f max( const Vector3f& v1, const Vector3f& v2 ) {
    return { std::max( v1.x, v2.x), std::max( v1.y, v2.y), std::max( v1.z, v2.z) };
}