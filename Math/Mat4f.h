#pragma once
#include "Vector4f.h"
class Mat4f {
public:
    __host__ __device__ Mat4f();
    __host__ __device__ Mat4f( const Vector4f& vec1, const Vector4f& vec2, const Vector4f& vec3, const Vector4f& vec4 );
    __host__ __device__ Vector4f& operator[]( int index );
    __host__ __device__ const Vector4f& operator[]( int index ) const;
    [[nodiscard]] __host__ __device__ float getDet() const;
    [[nodiscard]] __host__ __device__ Mat4f transpose() const;
    [[nodiscard]] __host__ __device__ Mat4f inverse() const;
    [[nodiscard]] __host__ __device__ static Mat4f identity();
private:
    Vector4f columns[4];
};

