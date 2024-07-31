#pragma once
#include "Vector.h"
#include "Vector4f.h"
#include "Mat4f.h"
#include "Mat3f.h"
#include "Mat2f.h"

__host__ __device__ float dot( const Vector3f& p1, const Vector3f& p2 );

__host__ __device__ float dot( const Vector4f& p1, const Vector4f& p2 );

__host__ __device__ float getDistance( const Vector3f& p1, const Vector3f& p2 );

__host__ __device__ Vector4f operator*( const Mat4f& m, const Vector4f& v );

__host__ __device__ Mat4f operator*( float a, const Mat4f& m );

__host__ __device__ Mat4f operator*( const Mat4f& m, float a );

__host__ __device__ Mat4f operator/( const Mat4f& m, float a );

__host__ __device__ Mat3f operator/( const Mat3f& m, float a );

__host__ __device__ Mat2f operator/( const Mat2f& m, float a );

__host__ __device__ Vector4f operator*( const Vector4f& v, const Mat4f& m );

__host__ __device__ Mat4f operator*( const Mat4f& m1, const Mat4f& m2 );

__host__ __device__ Vector3f operator*( const Mat3f& m, const Vector3f& v );

__host__ __device__ Mat3f operator*( const Mat3f& m1, const Mat3f& m2 );

__host__ __device__ Mat3f operator*( const Mat3f& m1, float a );

__host__ __device__ Mat3f operator*( float a, const Mat3f& m1 );

__host__ __device__ Mat3f operator+( const Mat3f& m1, const Mat3f& m2 );

__host__ __device__ Vector3f min( const Vector3f& v1, const Vector3f& v2 );

__host__ __device__ Vector3f max( const Vector3f& v1, const Vector3f& v2 );