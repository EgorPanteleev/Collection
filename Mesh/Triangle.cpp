#include "Triangle.h"
#include "Utils.h"
#include <iostream>
//Triangle::

constexpr float EPSILON = std::numeric_limits<float>::epsilon();
constexpr float MAX = std::numeric_limits<float>::max();

Triangle::Triangle(): v1(), v2(), v3() {
    edge1 = v2 - v1;
    edge2 = v3 - v1;
    N = edge1.cross( edge2 ).normalize();
    origin = (v1 + v2 + v3) / 3;
}

Triangle::Triangle( const Vector3f& v1, const Vector3f& v2, const Vector3f& v3 ): v1( v1 ), v2( v2 ), v3( v3 ) {
    edge1 = v2 - v1;
    edge2 = v3 - v1;
    N = edge1.cross( edge2 ).normalize();
    origin = (v1 + v2 + v3) / 3;
}

void Triangle::rotate( const Vector3f& axis, float angle ) {
    Mat3f rotation = Mat3f::getRotationMatrix( axis, angle );
    v1 = rotation * v1;
    v2 = rotation * v2;
    v3 = rotation * v3;
    edge1 = v2 - v1;
    edge2 = v3 - v1;
    N = edge1.cross( edge2 ).normalize();
    origin = (v1 + v2 + v3) / 3;
}

void Triangle::move( const Vector3f& p ) {
    v1 = v1 + p;
    v2 = v2 + p;
    v3 = v3 + p;
    origin = (v1 + v2 + v3) / 3;
}

void Triangle::moveTo( const Vector3f& point ) {
    move( point - getOrigin() );
}

void Triangle::scale( float scaleValue ) {
    v1 = v1 * scaleValue;
    v2 = v2 * scaleValue;
    v3 = v3 * scaleValue;
    edge1 = v2 - v1;
    edge2 = v3 - v1;
    N = edge1.cross( edge2 ).normalize();
    origin = (v1 + v2 + v3) / 3;
}
void Triangle::scale( const Vector3f& scaleVec ) {
    v1 = { v1[0] * scaleVec[0], v1[1] * scaleVec[1], v1[2] * scaleVec[2] };
    v2 = { v2[0] * scaleVec[0], v2[1] * scaleVec[1], v2[2] * scaleVec[2] };
    v3 = { v3[0] * scaleVec[0], v3[1] * scaleVec[1], v3[2] * scaleVec[2] };
    edge1 = v2 - v1;
    edge2 = v3 - v1;
    N = edge1.cross( edge2 ).normalize();
    origin = (v1 + v2 + v3) / 3;
}

void Triangle::scaleTo( float scaleValue ) {
    BBoxData bbox = getBBox();
    Vector3f len = bbox.pMax - bbox.pMin;
    Vector3f cff = { scaleValue / len[0], scaleValue / len[1], scaleValue / len[2] };
    scale( cff );
}
void Triangle::scaleTo( const Vector3f& scaleVec ) {
    BBoxData bbox = getBBox();
    Vector3f len = bbox.pMax - bbox.pMin;
    Vector3f cff = { scaleVec[0] / len[0], scaleVec[1] / len[1], scaleVec[2] / len[2] };
    scale( cff );
}

BBoxData Triangle::getBBox() const {
    float minX = std::min( std::min( v1[0], v2[0] ), v3[0] );
    float maxX = std::max( std::max( v1[0], v2[0] ), v3[0] );
    float minY = std::min( std::min( v1[1], v2[1] ), v3[1] );
    float maxY = std::max( std::max( v1[1], v2[1] ), v3[1] );
    float minZ = std::min( std::min( v1[2], v2[2] ), v3[2] );
    float maxZ = std::max( std::max( v1[2], v2[2] ), v3[2] );
    return { Vector3f( minX, minY, minZ ), Vector3f( maxX, maxY, maxZ ) };
}

Vector3f Triangle::getOrigin() const {
    return origin;
}

bool Triangle::isContainPoint( const Vector3f& p ) const {
    float detT = (v2.getY() - v3.getY()) * (v1.getX() - v3.getX()) + (v3.getX() - v2.getX()) * (v1.getY() - v3.getY());
    float alpha = ((v2.getY() - v3.getY()) * (p.getX() - v3.getX()) + (v3.getX() - v2.getX()) * (p.getY() - v3.getY())) / detT;
    float beta = ((v3.getY() - v1.getY()) * (p.getX() - v3.getX()) + (v1.getX() - v3.getX()) * (p.getY() - v3.getY())) / detT;
    float gamma = 1.0f - alpha - beta;

    // Check if the point is inside the triangle
    return ( alpha >= 0.0f && alpha <= 1.0f &&
           beta >= 0.0f && beta <= 1.0f &&
           gamma >= 0.0f && gamma <= 1.0f &&
           p.getZ() >=std::min( std::min(v1.getZ(), v2.getZ()), v3.getZ() ) &&
           p.getZ() <= std::max(std::max(v1.getZ(), v2.getZ()), v3.getZ() ) );
}

float Triangle::intersectsWithRay( const Ray& ray ) const {
    Vector3f h = ray.direction.cross( edge2 );
    float a = dot(edge1, h);

    if ( a < EPSILON ) return MAX; // Ray is parallel to the triangle

    float f = 1.0f / a;
    Vector3f s = ray.origin - v1;
    float u = f * dot(s, h);

    if ( u < 0.0f || u > 1.0f ) return MAX;

    Vector3f q = s.cross( edge1 );
    float v = f * dot(ray.direction, q);

    if  ( v < 0.0f || u + v > 1.0f ) return MAX;

    float t = f * dot(edge2, q);

    if ( t < EPSILON ) return MAX;

    return t;
}

Vector3f Triangle::getNormal() const {
    return N;
}
