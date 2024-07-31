#pragma once
#include "Vector3f.h"
class Mat3f {
public:
    __host__ __device__ Mat3f();
    __host__ __device__ Mat3f( const Vector3f& vec1, const Vector3f& vec2, const Vector3f& vec3 );
    __host__ __device__ Vector3f& operator[]( int index );
    __host__ __device__ const Vector3f& operator[]( int index ) const;
    __host__ __device__ bool operator==( Mat3f& mat ) const;
    __host__ __device__ bool operator!=( Mat3f& mat ) const;
    [[nodiscard]] __host__ __device__ float getAlgExtension( int col, int row ) const;
    [[nodiscard]] __host__ __device__ float getDet() const;
    [[nodiscard]] __host__ __device__ Mat3f transpose() const;
    [[nodiscard]] __host__ __device__ Mat3f inverse() const;
    [[nodiscard]] __host__ __device__ Mat3f getUnion() const;
    [[nodiscard]] __host__ __device__ static Mat3f identity();
    [[nodiscard]] __host__ __device__ static Mat3f getRotationMatrix( const Vector3f& axis, float angle );
private:
    Vector3f columns[3];
};

