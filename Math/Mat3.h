//
// Created by auser on 10/20/24.
//

#ifndef MATH_MAT3_H
#define MATH_MAT3_H
#include "Vec3.h"

template<typename Type>
class Mat3 {
public:
    Mat3(): columns() {
    }

    Mat3(const Vec3<Type>& vec1, const Vec3<Type>& vec2, const Vec3<Type>& vec3 ) {
        columns[0] = vec1;
        columns[1] = vec2;
        columns[2] = vec3;
    }

    Mat3<Type>& operator+=( const Mat3<Type>& other ) {
        columns[0] += other.columns[0];
        columns[1] += other.columns[1];
        columns[2] += other.columns[2];
        return *this;
    }

    Mat3<Type> operator+( const Mat3<Type>& other ) {
        return { columns[0] + other.columns[0], columns[1] + other.columns[1], columns[2] + other.columns[2] };
    }

    Mat3<Type>& operator*=( const Mat3<Type>& other ) {
        columns[0] = *this * other.columns[0];
        columns[1] = *this * other.columns[1];
        columns[2] = *this * other.columns[2];
        return *this;
    }
    Mat3<Type> operator*( const Mat3<Type>& other ) const {
        return { *this * other.columns[0], *this * other.columns[1], *this * other.columns[2] };
    }

    Vec3<Type>& operator[]( int index ) {
        return columns[index];
    }

    const Vec3<Type>& operator[]( int index ) const {
        return columns[index];
    }

    bool operator==( const Mat3<Type>& mat ) const {
        for ( int i = 0; i < 3; ++i ) {
            for ( int j = 0; j < 3; ++j ) {
                if ( columns[i][j] != mat[i][j] ) return false;
            }
        }
        return true;
    }

    bool operator!=( const Mat3<Type>& other ) const {
        return !( *this == other );
    }

    Type getAlgExtension( int col, int row ) const {
        Mat3<Type> mat;
        double sign = ( ( col + row ) % 2 == 0 ) ? 1 : -1;
        for ( int i = 0, si = 0; i < 3; ++i ) {
            if ( i == col ) continue;
            for ( int j = 0, sj = 0; j < 3; ++j ) {
                if ( j == row ) continue;
                mat[si][sj] = columns[i][j];
                ++sj;
            }
            ++si;
        }
        return sign * mat.getDet();
    }

    Type getDet() const {
        return ( columns[0][0] * columns[1][1] * columns[2][2] +
                 columns[1][0] * columns[2][1] * columns[0][2] +
                 columns[0][1] * columns[1][2] * columns[2][0] -
                 columns[0][2] * columns[1][1] * columns[2][0] -
                 columns[0][1] * columns[1][0] * columns[2][2] -
                 columns[1][2] * columns[2][1] * columns[0][0]);
    }

    Mat3<Type> transpose() const {
        Mat3<Type> res;
        int i = 0;
        for ( auto c : columns ) {
            res[0][i] = c[0];
            res[1][i] = c[1];
            res[2][i] = c[2];
            i++;
        }
        return res;
    }

    Mat3<Type> inverse() const {
        Mat3<Type> res;
        double det = getDet();
        if ( det == 0 ) return {};
        res = getUnion().transpose() / det;
        return res;
    }

    Mat3<Type> getUnion() const {
        return { {getAlgExtension(0, 0), getAlgExtension(0, 1), getAlgExtension(0, 2) },
                 {getAlgExtension(1, 0), getAlgExtension(1, 1), getAlgExtension(1, 2) },
                 {getAlgExtension(2, 0), getAlgExtension(2, 1), getAlgExtension(2, 2) } };
    }

    static Mat3<Type> identity() {
        return { {1, 0, 0},
                 {0, 1, 0},
                 {0, 0, 1}
        };
    }

    static Mat3<Type> getRotationMatrix( const Vec3<Type>& axis, Type angle ) {
        Vec3<Type> normalizedAxis = axis.normalize();
        angle = ( angle * M_PI ) / 180;
        Mat3<Type> skewSymmetric = {
                {0                ,normalizedAxis[2],-normalizedAxis[1]},
                {-normalizedAxis[2],0                ,normalizedAxis[0]},
                {normalizedAxis[1],-normalizedAxis[0],0                }
        };
        Mat3<Type> rotation = identity() + std::sin(angle) * skewSymmetric + (1 - std::cos(angle)) * skewSymmetric * skewSymmetric;

        return rotation;
    }

    Vec3<Type> columns[3];
};

template<typename Type>
inline Mat3<Type> operator*( const Mat3<Type>& mat, const Type& a ) {
    return { mat.columns[0] * a, mat.columns[1] * a, mat.columns[2] * a };
}

template<typename Type>
inline Mat3<Type> operator*( const Type& a, const Mat3<Type>& mat ) {
    return { mat.columns[0] * a, mat.columns[1] * a, mat.columns[2] * a };
}

template<typename Type>
inline Mat3<Type> operator/( const Mat3<Type>& mat, const Type& a ) {
    return { mat.columns[0] / a, mat.columns[1] / a, mat.columns[2] / a };
}

template<typename Type>
inline Mat3<Type> operator/( const Type& a, const Mat3<Type>& mat ) {
    return { mat.columns[0] / a, mat.columns[1] / a, mat.columns[2] / a };
}

template<typename Type>
inline Vec3<Type> operator*( const Mat3<Type>& mat, const Vec3<Type>& vec ) {
    return { mat.columns[0][0] * vec[0] + mat.columns[1][0] * vec[1] + mat.columns[2][0] * vec[2],
             mat.columns[0][1] * vec[0] + mat.columns[1][1] * vec[1] + mat.columns[2][1] * vec[2],
             mat.columns[0][2] * vec[0] + mat.columns[1][2] * vec[1] + mat.columns[2][2] * vec[2] };
}

template<typename Type>
inline Vec3<Type> operator*( const Vec3<Type>& vec, const Mat3<Type>& mat ) {
    return { mat.columns[0][0] * vec[0] + mat.columns[0][1] * vec[1] + mat.columns[0][2] * vec[2],
             mat.columns[1][0] * vec[0] + mat.columns[1][1] * vec[1] + mat.columns[1][2] * vec[2],
             mat.columns[2][0] * vec[0] + mat.columns[2][1] * vec[1] + mat.columns[2][2] * vec[2] };
}

typedef Mat3<double> Mat3d;
typedef Mat3<double> Mat3f;
typedef Mat3<int> Mat3i;


#endif //MATH_MAT3_H
