//
// Created by auser on 5/12/24.
//

#include "SphereMesh.h"
#include <cmath>
#include "Utils.h"
__host__ __device__ SphereMesh::SphereMesh(): radius(0), origin() {
}
__host__ __device__ SphereMesh::SphereMesh( float r, const Vector3f& pos ): radius(r), origin(pos) {
}

__host__ __device__ SphereMesh::SphereMesh( float r, const Vector3f& pos, const Material& m ): radius(r), origin(pos) {
    material = m;
}

__host__ __device__ void SphereMesh::rotate( const Vector3f& axis, float angle ) {
    Mat3f rotation = Mat3f::getRotationMatrix( axis, angle );
    Vector3f oldOrigin = origin;
    move( origin * ( -1 ));
    origin = rotation * origin;
    move( oldOrigin );
}

__host__ __device__ void SphereMesh::move( const Vector3f& p ) {
    origin = origin + p;
}

__host__ __device__ void SphereMesh::moveTo( const Vector3f& point ) {
    move( point - origin );
}

__host__ __device__ void SphereMesh::scale( float scaleValue ) {
    radius = radius * scaleValue;
}
__host__ __device__ void SphereMesh::scale( const Vector3f& scaleVec ) {
    if ( scaleVec[0] != scaleVec[1] || scaleVec[0] != scaleVec[2] ) return; //not sphere! nothing to do
    radius = radius * scaleVec[0];
}

__host__ __device__ void SphereMesh::scaleTo( float scaleValue ) {
    BBoxData bbox = getBBox();
    float len = bbox.pMax[0] - bbox.pMin[0];
    float cff = scaleValue / len;
    scale( cff );
}

__host__ __device__ void SphereMesh::scaleTo( const Vector3f& scaleVec ) {
    if ( scaleVec[0] != scaleVec[1] || scaleVec[0] != scaleVec[2] ) return; //not sphere! nothing to do
    BBoxData bbox = getBBox();
    float len = bbox.pMax[0] - bbox.pMin[0];
    float cff = scaleVec[0] / len;
    scale( cff );
}

__host__ __device__ BBoxData SphereMesh::getBBox() const {
    Vector3f r = { radius, radius, radius };
    return { origin - r, origin + r };
}

__host__ __device__ bool SphereMesh::isContainPoint( const Vector3f& p ) const {
    if ( getDistance( p, origin ) == radius ) return true;
    return false;
}

__host__ __device__ IntersectionData SphereMesh::intersectsWithRay( const Ray& ray ) const {
    Vector3f D = ray.direction;
    Vector3f OC = ray.origin - origin;
    float k1 = dot( D, D );
    float k2 = 2 * dot( OC, D );
    float k3 = dot( OC, OC ) - radius * radius;

    float disc = k2 * k2 - 4 * k1 * k3;
    if ( disc < 0 ) {
        return {};
    }
    disc = sqrt( disc ) / ( 2 * k1 );
    k2 = -k2 / ( 2 * k1 );
    float t1 = k2 + disc;
    float t2 = k2 - disc;
    if ( t1 < t2 ) t2 = t1;
    Vector3f P = ray.origin + ray.direction * t2;
    return { t2, getNormal( P ) , nullptr};
}

__host__ __device__ Vector3f SphereMesh::getNormal( const Vector3f& p ) const {
    return ( p - origin );
}

__host__ __device__ Vector3f SphereMesh::getOrigin() const {
    return origin;
}