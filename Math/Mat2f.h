#pragma once
#include "Vector2f.h"

class Mat2f {
public:
    __host__ __device__ Mat2f();
    __host__ __device__ Mat2f( const Vector2f& vec1, const Vector2f& vec2 );
    __host__ __device__ Vector2f& operator[]( int index );
    __host__ __device__ const Vector2f& operator[]( int index ) const;
    __host__ __device__ bool operator==( Mat2f& mat ) const;
    __host__ __device__ bool operator!=( Mat2f& mat ) const;
    [[nodiscard]] __host__ __device__ float getDet() const;
    [[nodiscard]] __host__ __device__ Mat2f transpose() const;
    [[nodiscard]] __host__ __device__ Mat2f inverse() const;
    [[nodiscard]] __host__ __device__ Mat2f getUnion() const;
private:
    Vector2f columns[3];
};


