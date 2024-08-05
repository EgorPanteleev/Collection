#pragma once
#include "Vector.h"
#include "Vector4f.h"
#include "Mat4f.h"
#include "Mat3f.h"
#include "Mat2f.h"
#include "Color.h"
float dot( const Vector3f& p1, const Vector3f& p2 );

float dot( const Vector4f& p1, const Vector4f& p2 );

float getDistance( const Vector3f& p1, const Vector3f& p2 );

Vector4f operator*( const Mat4f& m, const Vector4f& v );

Mat4f operator*( float a, const Mat4f& m );

Mat4f operator*( const Mat4f& m, float a );

Mat4f operator/( const Mat4f& m, float a );

Mat3f operator/( const Mat3f& m, float a );

Mat2f operator/( const Mat2f& m, float a );

Vector4f operator*( const Vector4f& v, const Mat4f& m );

Mat4f operator*( const Mat4f& m1, const Mat4f& m2 );

Vector3f operator*( const Mat3f& m, const Vector3f& v );

Vector3f operator*( float a, const Vector3f& v );

Vector3f operator*( float a, const RGB& col );

Mat3f operator*( const Mat3f& m1, const Mat3f& m2 );

Mat3f operator*( const Mat3f& m1, float a );

Mat3f operator*( float a, const Mat3f& m1 );

Mat3f operator+( const Mat3f& m1, const Mat3f& m2 );

RGB operator+( const RGB& col, const Vector3f& vec );

Vector3f min( const Vector3f& v1, const Vector3f& v2 );

Vector3f max( const Vector3f& v1, const Vector3f& v2 );