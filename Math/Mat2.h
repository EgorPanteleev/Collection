//
// Created by auser on 10/20/24.
//

#ifndef MATH_MAT2_H
#define MATH_MAT2_H
#include "Vec2.h"

template<typename Type>
class Mat2 {
public:
    Mat2(): columns() {
    }

    Mat2(const Vec2<Type>& vec1, const Vec2<Type>& vec2 ) {
        columns[0] = vec1;
        columns[1] = vec2;
    }

    Mat2<Type>& operator+=( const Mat2<Type>& other ) {
        columns[0] += other.columns[0];
        columns[1] += other.columns[1];
        return *this;
    }

    Mat2<Type> operator+( const Mat2<Type>& other ) {
        return { columns[0] + other.columns[0], columns[1] + other.columns[1] };
    }

    Mat2<Type>& operator*=( const Mat2<Type>& other ) {
        columns[0] = *this * other.columns[0];
        columns[1] = *this * other.columns[1];
        return *this;
    }
    Mat2<Type> operator*( const Mat2<Type>& other ) const {
        return { *this * other.columns[0], *this * other.columns[1] };
    }

    Vec2<Type>& operator[]( int index ) {
        return columns[index];
    }

    const Vec2<Type>& operator[]( int index ) const {
        return columns[index];
    }

    bool operator==( const Mat2<Type>& mat ) const {
        for ( int i = 0; i < 2; ++i ) {
            for ( int j = 0; j < 2; ++j ) {
                if ( columns[i][j] != mat[i][j] ) return false;
            }
        }
        return true;
    }

    bool operator!=( const Mat2<Type>& other ) const {
        return !( *this == other );
    }

    Type getDet() const {
        return ( columns[0][0] * columns[1][1] -
                 columns[1][0] * columns[0][1] );
    }

    Mat2<Type> transpose() const {
        Mat2<Type> res;
        int i = 0;
        for ( auto c : columns ) {
            res[0][i] = c[0];
            res[1][i] = c[1];
            i++;
        }
        return res;
    }

    Mat2<Type> inverse() const {
        Mat2<Type> res;
        double det = getDet();
        if ( det == 0 ) return {};
        res = getUnion().transpose() / det;
        return res;
    }

    Mat2<Type> getUnion() const {
        return {
                { columns[0][0], -columns[0][1] },
                { -columns[1][0], columns[1][1] }
        };
    }

    static Mat2<Type> identity() {
        return { { 1, 0 },
                 { 0, 1 }
        };
    }

    Vec2<Type> columns[2];
};

template<typename Type>
Mat2<Type> operator*( const Mat2<Type>& mat, const Type& a ) {
    return { mat.columns[0] * a, mat.columns[1] * a };
}

template<typename Type>
Mat2<Type> operator*( const Type& a, const Mat2<Type>& mat ) {
    return { mat.columns[0] * a, mat.columns[1] * a };
}

template<typename Type>
Mat2<Type> operator/( const Mat2<Type>& mat, const Type& a ) {
    return { mat.columns[0] / a, mat.columns[1] / a };
}

template<typename Type>
Mat2<Type> operator/( const Type& a, const Mat2<Type>& mat ) {
    return { mat.columns[0] / a, mat.columns[1] / a };
}

template<typename Type>
Vec2<Type> operator*( const Mat2<Type>& mat, const Vec2<Type>& vec ) {
    return { mat.columns[0][0] * vec[0] + mat.columns[1][0] * vec[1],
             mat.columns[0][1] * vec[0] + mat.columns[1][1] * vec[1] };
}

template<typename Type>
Vec2<Type> operator*( const Vec2<Type>& vec, const Mat2<Type>& mat ) {
    return { mat.columns[0][0] * vec[0] + mat.columns[0][1] * vec[1],
             mat.columns[1][0] * vec[0] + mat.columns[1][1] * vec[1] };
}

typedef Mat2<double> Mat2d;
typedef Mat2<double> Mat2f;
typedef Mat2<int> Mat2i;

#endif //MATH_MAT2_H
