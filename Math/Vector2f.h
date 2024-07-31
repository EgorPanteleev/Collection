#pragma once
#include "hip/hip_runtime.h"
class Vector2f {
public:
    __host__ __device__ void setX( float _x );

    __host__ __device__ void setY( float _y );

    [[nodiscard]] __host__ __device__ float getX() const;

    [[nodiscard]] __host__ __device__ float getY() const;

    __host__ __device__ void set( const Vector2f& p );

    __host__ __device__ Vector2f& operator=( const Vector2f& p );

    [[nodiscard]] __host__ __device__ Vector2f normalize() const;

    __host__ __device__ float& operator[]( int index );

    __host__ __device__ const float& operator[]( int index ) const;

    __host__ __device__ Vector2f operator+( const Vector2f& p ) const;

    __host__ __device__ Vector2f operator-( const Vector2f& p ) const;

    __host__ __device__ Vector2f operator*( float a ) const;

    __host__ __device__ Vector2f operator/( float a ) const;

    __host__ __device__ bool operator==( const Vector2f& p ) const;

    __host__ __device__ bool operator!=( const Vector2f& p ) const;
    __host__ __device__ Vector2f();
    __host__ __device__ Vector2f(float _x, float _y );
    __host__ __device__ ~Vector2f();
    __host__ __device__ Vector2f( const Vector2f& p );
private:
    float x{};
    float y{};
};

