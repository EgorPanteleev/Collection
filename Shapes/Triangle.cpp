#include "Triangle.h"
#include "Utils.h"
#include <iostream>
//Triangle::
Triangle::Triangle(): v1(), v2(), v3() {
    Vector3f edge1 = v2 - v1;
    Vector3f edge2 = v3 - v2;
    N = edge1.cross( edge2 ).normalize();
}

Triangle::Triangle( const Vector3f& v1, const Vector3f& v2, const Vector3f& v3 ): v1( v1 ), v2( v2 ), v3( v3 ) {
    Vector3f edge1 = v2 - v1;
    Vector3f edge2 = v3 - v1;
    N = edge1.cross( edge2 ).normalize();
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
    Vector3f edge1 = v2 - v1;
    Vector3f edge2 = v3 - v1;
    Vector3f h = ray.getDirection().cross( edge2 );
    float a = dot(edge1, h);

    if (a > -std::numeric_limits<float>::epsilon() && a < std::numeric_limits<float>::epsilon())
        return std::numeric_limits<float>::max(); // Ray is parallel to the triangle

    float f = 1.0f / a;
    Vector3f s = ray.getOrigin() - v1;
    float u = f * dot(s, h);

    if (u < 0.0f || u > 1.0f)
        return std::numeric_limits<float>::max();

    Vector3f q = s.cross( edge1 );
    float v = f * dot(ray.getDirection(), q);

    if (v < 0.0f || u + v > 1.0f)
        return std::numeric_limits<float>::max();

    float t = f * dot(edge2, q);

    if ( t < std::numeric_limits<float>::epsilon() ) return std::numeric_limits<float>::max();

    return t;
}
Vector3f Triangle::getNormal() const {
    return N;
}