#pragma once
#include "hip/hip_runtime.h"
class Vector3f {
public:
    __host__ __device__ void setX( float _x );

    __host__ __device__ void setY( float _y );

    __host__ __device__ void setZ( float _z );

    [[nodiscard]] __host__ __device__ float getX() const;

    [[nodiscard]] __host__ __device__ float getY() const;

    [[nodiscard]] __host__ __device__ float getZ() const;

    __host__ __device__ void set( const Vector3f& p );
    //operators

    __host__ __device__ Vector3f& operator=( const Vector3f& p );

    [[nodiscard]] __host__ __device__ Vector3f normalize() const;

    [[nodiscard]] __host__ __device__ Vector3f cross( const Vector3f& vec ) const;

    __host__ __device__ float& operator[]( int index );

    __host__ __device__ const float& operator[]( int index ) const;

    __host__ __device__ Vector3f operator+( const Vector3f& p ) const;

    __host__ __device__ Vector3f operator-( const Vector3f& p ) const;

    __host__ __device__ Vector3f operator*( float a ) const;

    __host__ __device__ Vector3f operator/( float a ) const;

    __host__ __device__ bool operator==( const Vector3f& p ) const;

    __host__ __device__ bool operator!=( const Vector3f& p ) const;
    __host__ __device__ Vector3f();
    __host__ __device__ Vector3f(float _x, float _y, float _z);
    __host__ __device__ ~Vector3f();
    __host__ __device__ Vector3f( const Vector3f& p );

    float x;
    float y;
    float z;
};
