#pragma once
#include "hip/hip_runtime.h"
class Vector4f {
public:
    __host__ __device__ void setX( float _x );

    __host__ __device__ void setY( float _y );

    __host__ __device__ void setZ( float _z );

    __host__ __device__ void setW( float _w );

    [[nodiscard]] __host__ __device__ float getX() const;

    [[nodiscard]] __host__ __device__ float getY() const;

    [[nodiscard]] __host__ __device__ float getZ() const;

    [[nodiscard]] __host__ __device__ float getW() const;

    __host__ __device__ void set( const Vector4f& p );

    __host__ __device__ Vector4f& operator=( const Vector4f& p );

    [[nodiscard]] __host__ __device__ Vector4f normalize() const;

    [[nodiscard]] __host__ __device__ Vector4f cross( Vector4f vec ) const;

    __host__ __device__ float& operator[]( int index );

    __host__ __device__ const float& operator[]( int index ) const;

    __host__ __device__ Vector4f operator+( const Vector4f& p ) const;

    __host__ __device__ Vector4f operator-( const Vector4f& p ) const;

    __host__ __device__ Vector4f operator*( float a ) const;

    __host__ __device__ Vector4f operator/( float a ) const;

    __host__ __device__ bool operator==( const Vector4f& p ) const;

    __host__ __device__ bool operator!=( const Vector4f& p ) const;
    __host__ __device__ Vector4f();
    __host__ __device__ Vector4f(float _x, float _y, float _z, float _w);
    __host__ __device__ ~Vector4f();
    __host__ __device__ Vector4f( const Vector4f& p );
private:
    float x{};
    float y{};
    float z{};
    float w{};
};



