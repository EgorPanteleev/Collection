#include "Mat2f.h"
#include "Utils.h"
__host__ __device__ Vector2f& Mat2f::operator[]( int index ) {
    return columns[index];
}

__host__ __device__ const Vector2f& Mat2f::operator[]( int index ) const {
    return columns[index];
}

__host__ __device__ bool Mat2f::operator==( Mat2f& mat ) const {
    for ( int i = 0; i < 2; ++i ) {
        for ( int j = 0; j < 2; ++j ) {
            if ( columns[i][j] != mat[i][j] ) return false;
        }
    }
    return true;
}

__host__ __device__ bool Mat2f::operator!=( Mat2f& mat ) const {
    return ( ! (*this == mat) );
}
__host__ __device__ Mat2f::Mat2f(): columns() { }

__host__ __device__ Mat2f::Mat2f( const Vector2f& vec1, const Vector2f& vec2 ) {
    columns[0] = vec1;
    columns[1] = vec2;
}

__host__ __device__ float Mat2f::getDet() const {
    return ( columns[0][0] * columns[1][1] -
             columns[1][0] * columns[0][1] );
}

__host__ __device__ Mat2f Mat2f::transpose() const {
    Mat2f res;
    int i = 0;
    for ( auto c : columns ) {
        res[0][i] = c[0];
        res[1][i] = c[1];
        i++;
    }
    return res;
}

__host__ __device__ Mat2f Mat2f::inverse() const {
    Mat2f res;
    float det = getDet();
    if ( det == 0 ) return {};
    res = getUnion().transpose() / det;
    return res;
}

__host__ __device__ Mat2f Mat2f::getUnion() const {
    return {
            { columns[0][0], -columns[0][1] },
            { -columns[1][0], columns[1][1] }
    };
}