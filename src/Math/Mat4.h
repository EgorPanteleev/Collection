//
// Created by auser on 10/20/24.
//

#ifndef MATH_MAT4_H
#define MATH_MAT4_H
#include "Vec4.h"
#include "Mat3.h"

template<typename Type>
class Mat4 {
public:
    Mat4(): columns() {
    }

    Mat4(const Vec4<Type>& vec1, const Vec4<Type>& vec2, const Vec4<Type>& vec3, const Vec4<Type>& vec4 ) {
        columns[0] = vec1;
        columns[1] = vec2;
        columns[2] = vec3;
        columns[3] = vec4;
    }

    Mat3<Type>& operator+=( const Mat3<Type>& other ) {
        columns[0] += other.columns[0];
        columns[1] += other.columns[1];
        columns[2] += other.columns[2];
        columns[3] += other.columns[3];
        return *this;
    }

    Mat3<Type> operator+( const Mat3<Type>& other ) {
        return { columns[0] + other.columns[0], columns[1] + other.columns[1],
                 columns[2] + other.columns[2], columns[3] + other.columns[3] };
    }

    Mat4<Type>& operator*=( const Mat4<Type>& other ) {
        columns[0] = *this * other.columns[0];
        columns[1] = *this * other.columns[1];
        columns[2] = *this * other.columns[2];
        columns[3] = *this * other.columns[3];
        return *this;
    }
    Mat4<Type> operator*( const Mat4<Type>& other ) const {
        return { *this * other.columns[0], *this * other.columns[1],
                 *this * other.columns[2], *this * other.columns[3] };
    }

    Vec4<Type>& operator[]( int index ) {
        return columns[index];
    }

    const Vec4<Type>& operator[]( int index ) const {
        return columns[index];
    }

    bool operator==( const Mat4<Type>& mat ) const {
        for ( int i = 0; i < 4; ++i ) {
            for ( int j = 0; j < 4; ++j ) {
                if ( columns[i][j] != mat[i][j] ) return false;
            }
        }
        return true;
    }

    bool operator!=( const Mat4<Type>& other ) const {
        return !( *this == other );
    }

    Type getDet() const {
        Mat3<Type> first  = Mat3<Type>(Vec3<Type>(columns[0][1],columns[0][2] ,columns[0][3]),
                                       Vec3<Type>(columns[1][1],columns[1][2] ,columns[1][3]),
                                       Vec3<Type>(columns[2][1],columns[2][2] ,columns[2][3]));
        Mat3<Type> second = Mat3<Type>(Vec3<Type>(columns[0][0],columns[0][2] ,columns[0][3]),
                                       Vec3<Type>(columns[1][0],columns[1][2] ,columns[1][3]),
                                       Vec3<Type>(columns[2][0],columns[2][2] ,columns[2][3]));
        Mat3<Type> third  = Mat3<Type>(Vec3<Type>(columns[0][0],columns[0][1] ,columns[0][3]),
                                       Vec3<Type>(columns[1][0],columns[1][1] ,columns[1][3]),
                                       Vec3<Type>(columns[2][0],columns[2][1] ,columns[2][3]));
        Mat3<Type> fourth = Mat3<Type>(Vec3<Type>(columns[0][0],columns[0][1] ,columns[0][2]),
                                       Vec3<Type>(columns[1][0],columns[1][1] ,columns[1][2]),
                                       Vec3<Type>(columns[2][0],columns[2][1] ,columns[2][2]));
        return (-columns[3][0] * first.getDet() + columns[3][1] * second.getDet() -
                 columns[3][2] * third.getDet() + columns[3][3] * fourth.getDet());
    }

    Mat4<Type> transpose() const {
        Mat4<Type> res;
        int i = 0;
        for ( auto c : columns ) {
            res[0][i] = c[0];
            res[1][i] = c[1];
            res[2][i] = c[2];
            res[3][i] = c[2];
            i++;
        }
        return res;
    }

    Mat4<Type> inverse() const {
        return transpose() / getDet();
    }

    static Mat4<Type> identity() {
        return {
                { 1, 0, 0, 0 },
                { 0, 1, 0, 0 },
                { 0, 0, 1, 0 },
                { 0, 0, 0, 1 }
        };
    }

    static Mat4<Type> translate(const Vec4<Type>& translation) {
        return
        {
            { 1, 0, 0, translation.x },
            { 0, 1, 0, translation.y },
            { 0, 0, 1, translation.z },
            { 0, 0, 0, 1 }
        };
    }

    static Mat4<Type> rotateY(Type angleRad ) {
        Type cosAngle = cos(angleRad);
        Type sinAngle = sin(angleRad);
        return
        {
            { cosAngle , 0, sinAngle, 0 },
            { 0        , 1, 0       , 0 },
            { -sinAngle, 0, cosAngle, 0 },
            { 0        , 0, 0       , 1 }
        };
    }

    static Mat4<Type> rotateX(Type angleRad) {
        Type cosAngle = cos(angleRad);
        Type sinAngle = sin(angleRad);
        return
        {
            { 1, 0       , 0        , 0 },
            { 0, cosAngle, -sinAngle, 0 },
            { 0, sinAngle, cosAngle , 0 },
            { 0, 0       , 0        , 1 }
        };
    }

    static Mat4<Type> rotateZ(Type angleRad) {
        Type cosAngle = cos(angleRad);
        Type sinAngle = sin(angleRad);
        return
        {
            { cosAngle, -sinAngle, 0, 0 },
            { sinAngle, cosAngle , 0, 0 },
            { 0       , 0        , 1, 0 },
            { 0       , 0        , 0, 1 }
        };
    }

    Vec4<Type> columns[4];
};

template<typename Type>
inline Mat4<Type> operator*( const Mat4<Type>& mat, const Type& a ) {
    return { mat.columns[0] * a, mat.columns[1] * a, mat.columns[2] * a, mat.columns[3] * a };
}

template<typename Type>
inline Mat4<Type> operator*( const Type& a, const Mat4<Type>& mat ) {
    return { mat.columns[0] * a, mat.columns[1] * a, mat.columns[2] * a, mat.columns[3] * a };
}

template<typename Type>
inline Mat4<Type> operator/( const Mat4<Type>& mat, const Type& a ) {
    return { mat.columns[0] / a, mat.columns[1] / a, mat.columns[2] / a, mat.columns[3] / a };
}

template<typename Type>
inline Mat4<Type> operator/( const Type& a, const Mat4<Type>& mat ) {
    return { mat.columns[0] / a, mat.columns[1] / a, mat.columns[2] / a, mat.columns[3] / a };
}

template<typename Type>
inline Vec4<Type> operator*( const Mat4<Type>& mat, const Vec4<Type>& vec ) {
    return { mat.columns[0][0] * vec[0] + mat.columns[1][0] * vec[1] + mat.columns[2][0] * vec[2] + mat.columns[3][0] * vec[3],
             mat.columns[0][1] * vec[0] + mat.columns[1][1] * vec[1] + mat.columns[2][1] * vec[2] + mat.columns[3][1] * vec[3],
             mat.columns[0][2] * vec[0] + mat.columns[1][2] * vec[1] + mat.columns[2][2] * vec[2] + mat.columns[3][2] * vec[3],
             mat.columns[0][3] * vec[0] + mat.columns[1][3] * vec[1] + mat.columns[2][3] * vec[2] + mat.columns[3][3] * vec[3] };
}

template<typename Type>
inline Vec4<Type> operator*( const Vec4<Type>& vec, const Mat4<Type>& mat ) {
    return { mat.columns[0][0] * vec[0] + mat.columns[0][1] * vec[1] + mat.columns[0][2] * vec[2] + mat.columns[0][3] * vec[3],
             mat.columns[1][0] * vec[0] + mat.columns[1][1] * vec[1] + mat.columns[1][2] * vec[2] + mat.columns[1][3] * vec[3],
             mat.columns[2][0] * vec[0] + mat.columns[2][1] * vec[1] + mat.columns[2][2] * vec[2] + mat.columns[2][3] * vec[3],
             mat.columns[3][0] * vec[0] + mat.columns[3][1] * vec[1] + mat.columns[3][2] * vec[2] + mat.columns[3][3] * vec[3] };
}

typedef Mat4<double> Mat4d;
typedef Mat4<double> Mat4f;
typedef Mat4<int> Mat4i;

#endif //MATH_MAT4_H
